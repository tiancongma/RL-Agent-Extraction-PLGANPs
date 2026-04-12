#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import ast
import csv
import json
import os
import re
import shutil
import sys
import time
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
PROMPT_PREVIEW_NAME = "stage2_prompt_preview_v1.tsv"
TABLE_SELECTION_DEBUG_NAME = "table_selection_debug_v1.json"
CANDIDATE_SEGMENTATION_DEBUG_NAME = "candidate_segmentation_debug_v1.tsv"
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
GENERAL_SELECTOR_PROFILE = "role_aware_general_v1"
DOE_SELECTOR_OVERLAY = "doe_optimization_v1"
GENERAL_REQUIRED_ROLES = [
    "PREPARATION_METHOD",
    "MATERIALS",
    "FORMULATION_TABLE",
    "FORMULATION_RESULT",
    "OPTIMIZATION_RESULT",
    "CONTEXT_FALLBACK",
]
DOE_REQUIRED_ROLES = [
    "PREPARATION_METHOD",
    "MATERIALS",
    "EXPERIMENTAL_DESIGN",
    "VARIABLE_TABLE",
    "FORMULATION_TABLE",
    "OPTIMIZATION_RESULT",
    "CONTEXT_FALLBACK",
]
SECONDARY_ELIGIBLE_ROLES: set[str] = set()
ROLE_THRESHOLD_BY_ROLE = {
    "PREPARATION_METHOD": 6.0,
    "MATERIALS": 6.0,
    "EXPERIMENTAL_DESIGN": 5.0,
    "VARIABLE_TABLE": 5.0,
    "FORMULATION_TABLE": 5.0,
    "FORMULATION_RESULT": 4.0,
    "OPTIMIZATION_RESULT": 4.0,
    "CONTEXT_FALLBACK": 1.0,
}
SECONDARY_THRESHOLD_BONUS_BY_ROLE = {
    "FORMULATION_TABLE": 4.0,
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
ROLE_HEADING_CUES = {
    "PREPARATION_METHOD": [
        "preparation",
        "nanoparticles preparation",
        "formulation",
        "method",
        "synthesis",
        "nanoprecipitation",
        "emulsion solvent evaporation",
    ],
    "MATERIALS": [
        "materials",
        "chemicals",
        "reagents",
        "materials and methods",
    ],
    "EXPERIMENTAL_DESIGN": [
        "experimental design",
        "box-behnken",
        "response surface",
        "rsm",
        "design expert",
    ],
    "FORMULATION_RESULT": [
        "results and discussion",
        "effect of independent variables",
        "characterization",
    ],
    "OPTIMIZATION_RESULT": [
        "optimized formulation",
        "optimization",
        "point prediction",
        "validity of model",
    ],
}
ROLE_LEXICAL_CUES = {
    "PREPARATION_METHOD": [
        "dissolved",
        "organic phase",
        "aqueous phase",
        "added dropwise",
        "stirring",
        "stirred",
        "evaporation",
        "evaporated",
        "centrifuged",
        "washed",
        "redispersed",
        "nanoprecipitation",
        "emulsion solvent evaporation",
        "prepared by",
        "formulations were prepared",
    ],
    "MATERIALS": [
        "purchased from",
        "obtained from",
        "sigma",
        "aldrich",
        "fisher",
        "hplc grade",
        "molecular weight",
        "plga",
        "poloxamer",
        "acetone",
        "water",
        "resomer",
    ],
    "EXPERIMENTAL_DESIGN": [
        "experimental design",
        "box-behnken",
        "response surface",
        "rsm",
        "design expert",
        "polynomial model",
        "independent variables",
        "dependent variables",
        "coded value",
        "desirability",
    ],
    "VARIABLE_TABLE": [
        "independent variables",
        "dependent variables",
        "levels",
        "factors",
        "responses",
        "coded levels",
        "box-behnken",
        "design",
        "different concentrations",
        "different ratio",
        "different ratios",
        "initial ratios",
    ],
    "FORMULATION_TABLE": [
        "formulation",
        "runs",
        "process variables",
        "dependent variable",
        "particle size",
        "entrapment",
        "pdi",
        "zeta potential",
        "drug entrapment",
    ],
    "FORMULATION_RESULT": [
        "particle size",
        "zeta potential",
        "drug entrapment",
        "formulation",
        "results and discussion",
    ],
    "OPTIMIZATION_RESULT": [
        "optimized formulation",
        "optimization",
        "point prediction",
        "desirability",
        "predicted values",
        "experimental values",
        "validity of model",
        "drug loading",
        "selected by applying constraints",
    ],
    "CONTEXT_FALLBACK": [
        "nanoparticles",
        "plga",
        "formulation",
        "drug entrapment",
        "particle size",
    ],
}
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


def evidence_blocks_path(out_dir: Path, key: str) -> Path:
    return out_dir / EVIDENCE_BLOCKS_SUBDIR / key / EVIDENCE_BLOCKS_FILENAME


def candidate_blocks_path(out_dir: Path, key: str) -> Path:
    return out_dir / CANDIDATE_BLOCKS_SUBDIR / key / CANDIDATE_BLOCKS_FILENAME


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


def call_gemini(model: str, prompt: str, retries: int, sleep_sec: float, *, progress_label: str = "") -> str:
    ensure_genai(model)
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            mdl = genai.GenerativeModel(model)
            resp = mdl.generate_content(
                prompt,
                generation_config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                },
            )
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
        if attempt < retries:
            if progress_label:
                print(
                    f"{progress_label} retrying attempt={attempt + 2}/{retries + 1}",
                    flush=True,
                )
            time.sleep(sleep_sec)
    raise last_err or RuntimeError("Gemini call failed.")


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


def paragraph_formulation_table_score(candidate: dict[str, Any]) -> dict[str, Any]:
    text = normalize_text(candidate.get("text_content"))
    lower = text.lower()
    split_trigger = normalize_text(candidate.get("split_trigger")).lower()
    quality_flags = {normalize_text(flag).lower() for flag in ensure_list(candidate.get("quality_flags"))}
    cue_score = 0.0
    structure_score = 0.0
    penalty_score = 0.0
    if "table " in lower:
        structure_score += 2.0
    if split_trigger in {"table_inline_split", "post_table_split", "inline_table_recovery"}:
        structure_score += 2.0
    if has_strong_formulation_table_signal(lower):
        cue_score += 5.0
    if "inline_formulation_table_recovered" in quality_flags:
        structure_score += 4.0
    if is_obvious_figure_or_front_matter_table(lower):
        penalty_score -= 6.0
    final_score = cue_score + structure_score + penalty_score
    return {
        "heading_score": 0.0,
        "cue_score": cue_score,
        "structure_score": structure_score,
        "penalty_score": penalty_score,
        "final_score": final_score,
    }


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
    ]
    has_fixed = any(re.search(pattern, lower) for pattern in fixed_patterns)
    has_varying = any(re.search(pattern, lower) for pattern in varying_patterns)
    return has_fixed and has_varying


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


def render_full_table_block(path: Path, *, max_lines_per_table: int = 24) -> str:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for idx, row in enumerate(reader):
            if idx >= max_lines_per_table:
                break
            lines.append(" | ".join(normalize_text(cell) for cell in row if normalize_text(cell)))
    if not lines:
        return ""
    return f"[TABLE {to_repo_rel(path)}]\n" + "\n".join(lines)


def table_mode() -> str:
    mode = normalize_text(os.getenv("STAGE2_TABLE_MODE", "full")).lower()
    return mode if mode in {"full", "summary"} else "full"


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
            }
        )
    entries.sort(
        key=lambda item: (
            -int(item["score"]),
            str(item["path"].name).lower(),
        )
    )
    return entries


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
        path = item["path"]
        rows = item["rows"]
        meta = item["meta"]
        header_row = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []
        header_parts: list[str] = []
        for cell in header_row:
            header_parts.extend(parse_header_cell(cell))
        header_parts = [part for part in header_parts if part]
        units = infer_units_from_headers(header_parts)
        row_ids = [row[0] for row in data_rows if row and normalize_text(row[0])]
        samples = select_sample_rows(data_rows)
        sample_lines = []
        for idx, row in enumerate(samples, start=1):
            sample_lines.append(f"- sample_row_{idx}: " + " | ".join(cell for cell in row if cell))
        rel = path.resolve().relative_to(PROJECT_ROOT).as_posix()
        table_match = re.search(r"__table_(\d+)__", path.name)
        table_id = f"Table {int(table_match.group(1))}" if table_match else path.stem
        caption = normalize_text(meta.get("caption_or_title"))
        footnotes = normalize_text(meta.get("footnotes") or meta.get("notes"))
        row_anchor_preview = summarize_row_anchor_preview(row_ids) if enhancement_enabled else ""
        block_lines = [
            f"[TABLE_SUMMARY {rel}]",
            f"- table_id: {table_id}",
            f"- title_or_caption: {caption or '(not available)'}",
            f"- column_headers: {' || '.join(header_parts) if header_parts else '(not available)'}",
            f"- header_units: {', '.join(units) if units else '(none inferred)'}",
            f"- row_identifier_pattern: {infer_row_pattern(row_ids)}",
            f"- table_role_hint: {infer_table_role_hint(header_parts, meta)}",
            f"- table_shape_hint: rows={meta.get('n_rows', len(rows))}, cols={meta.get('n_cols', max((len(r) for r in rows), default=0))}",
        ]
        if enhancement_enabled and row_anchor_preview:
            block_lines.append(f"- first_column_row_labels_preview: {row_anchor_preview}")
        if sample_lines:
            block_lines.append("- sample_rows:")
            block_lines.extend(sample_lines)
        block_lines.append(f"- footnotes_or_notes: {footnotes or '(not available)'}")
        blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks)


def render_summary_table_block(item: dict[str, Any], *, enhancement_enabled: bool) -> str:
    path = item["path"]
    rows = item["rows"]
    meta = item["meta"]
    header_row = rows[0]
    data_rows = rows[1:] if len(rows) > 1 else []
    header_parts: list[str] = []
    for cell in header_row:
        header_parts.extend(parse_header_cell(cell))
    header_parts = [part for part in header_parts if part]
    units = infer_units_from_headers(header_parts)
    row_ids = [row[0] for row in data_rows if row and normalize_text(row[0])]
    samples = select_sample_rows(data_rows)
    sample_lines = []
    for idx, row in enumerate(samples, start=1):
        sample_lines.append(f"- sample_row_{idx}: " + " | ".join(cell for cell in row if cell))
    table_match = re.search(r"__table_(\d+)__", path.name)
    table_id = f"Table {int(table_match.group(1))}" if table_match else path.stem
    caption = normalize_text(meta.get("caption_or_title"))
    footnotes = normalize_text(meta.get("footnotes") or meta.get("notes"))
    row_anchor_preview = summarize_row_anchor_preview(row_ids) if enhancement_enabled else ""
    block_lines = [
        f"[TABLE_SUMMARY {to_repo_rel(path)}]",
        f"- table_id: {table_id}",
        f"- title_or_caption: {caption or '(not available)'}",
        f"- column_headers: {' || '.join(header_parts) if header_parts else '(not available)'}",
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
    return "\n".join(block_lines)


def resolved_selector_strategy(current_table_mode: str) -> str:
    if current_table_mode == "summary":
        return "score_ranked_top_k"
    return "sorted_csv_first_4"


def detect_pre_llm_signals(raw_text: str, table_candidates: list[dict[str, Any]]) -> dict[str, bool]:
    combined = normalize_text(raw_text).lower()
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
    return {
        "has_doe_signal": has_doe_signal,
        "has_sequential_signal": has_sequential_signal,
        "has_optimization_signal": has_optimization_signal,
    }


def count_cue_hits(text: str, cues: list[str]) -> int:
    lower = normalize_text(text).lower()
    return sum(1 for cue in cues if cue in lower)


def document_position_score(index: int, total: int) -> float:
    if total <= 1:
        return 1.0
    ratio = index / max(total - 1, 1)
    if ratio <= 0.2:
        return 2.0
    if ratio <= 0.5:
        return 1.0
    return 0.0


def looks_like_section_heading(text: str, role: str) -> bool:
    compact = normalize_text(text).lower()
    if role == "PREPARATION_METHOD":
        return bool(
            re.search(r"\b\d+(?:\.\d+)*\.?\s*(nanoparticles preparation|preparation|formulation|method|synthesis)\b", compact)
            or compact.startswith("materials and methods")
        )
    if role == "MATERIALS":
        return bool(re.search(r"\b\d+(?:\.\d+)*\.?\s*materials\b", compact) or compact.startswith("materials"))
    if role == "EXPERIMENTAL_DESIGN":
        return bool(re.search(r"\bexperimental design\b|\bbox-behnken\b|\bdesign expert\b", compact))
    if role == "OPTIMIZATION_RESULT":
        return bool(re.search(r"\boptimized formulation\b|\boptimization\b", compact))
    return False


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


def paragraph_role_score(entry: dict[str, Any], role: str, *, total_paragraphs: int) -> dict[str, Any]:
    text = entry["text"]
    lower = normalize_text(text).lower()
    heading_score = 4.0 if looks_like_section_heading(text, role) else float(count_cue_hits(lower, ROLE_HEADING_CUES.get(role, [])))
    cue_score = float(count_cue_hits(lower, ROLE_LEXICAL_CUES.get(role, [])))
    structure_score = document_position_score(int(entry.get("paragraph_index", 0)), total_paragraphs)
    penalty_score = 0.0
    if role == "PREPARATION_METHOD":
        penalty_score -= 2.0 * count_cue_hits(lower, PREPARATION_NEGATIVE_CUES)
        cue_score += 1.5 * procedure_signal_count(lower)
        if has_preparation_procedure_signal(lower):
            structure_score += 4.0
        penalty_score -= 2.5 * assay_comparator_signal_count(lower)
        if is_assay_comparator_dominated(lower):
            penalty_score -= 8.0
    if role == "MATERIALS":
        penalty_score -= 1.5 * count_cue_hits(lower, MATERIALS_NEGATIVE_CUES)
        cue_score += 2.0 * count_any_cues(lower, PROCUREMENT_CUES)
        if is_materials_inventory_candidate(lower):
            structure_score += 4.0
    if "references" in lower or "copyright" in lower or "correspondence" in lower:
        penalty_score -= 4.0
    if role == "FORMULATION_RESULT" and "results and discussion" in lower:
        structure_score += 1.0
    if role == "EXPERIMENTAL_DESIGN":
        if "results and discussion" in lower or "zeta potential analysis" in lower:
            penalty_score -= 2.0
    if role == "OPTIMIZATION_RESULT":
        if "optimized" in lower or "desirability" in lower:
            structure_score += 2.0
        if "predicted" in lower and "experimental" in lower:
            cue_score += 1.0
        if optimization_decision_signature(lower):
            structure_score += 4.0
        cue_score += 1.0 * count_any_cues(lower, OPTIMIZATION_DECISION_CUES)
    if role == "CONTEXT_FALLBACK":
        cue_score = max(cue_score, 1.0 if len(lower.split()) >= 40 else 0.0)
        if "references" in lower or "journal of" in lower or re.search(r"\[\d+\]", text):
            penalty_score -= 6.0
    final_score = heading_score + cue_score + structure_score + penalty_score
    return {
        "heading_score": heading_score,
        "cue_score": cue_score,
        "structure_score": structure_score,
        "penalty_score": penalty_score,
        "final_score": final_score,
    }


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


def table_role_score(item: dict[str, Any], role: str) -> dict[str, Any]:
    meta, rows, blob, quality_flags, row_labels, section_kind, raw_score = selector_table_context(item)
    heading_score = float(count_cue_hits(blob, ROLE_HEADING_CUES.get(role, [])))
    cue_score = float(count_cue_hits(blob, ROLE_LEXICAL_CUES.get(role, [])))
    structure_score = 0.0
    penalty_score = 0.0
    preview_starts_figure = "first_column_row_labels_preview: figure " in blob.lower()
    if role == "VARIABLE_TABLE":
        if any("levels" in label or "independent variables" in label for label in row_labels):
            structure_score += 3.0
        if any("dependent variables" in blob for _ in [0]):
            structure_score += 2.0
        if any(token in blob for token in ["different concentrations", "different ratios", "initial ratios"]):
            structure_score += 3.0
        if "noise_rows_filtered" in quality_flags and cue_score + structure_score < 5.0:
            penalty_score -= 2.0
    if role == "FORMULATION_TABLE":
        row_id_hits = sum(1 for label in row_labels if any(re.search(pattern, label, flags=re.I) for pattern in TABLE_ROW_ID_PATTERNS))
        structure_score += min(4.0, float(row_id_hits))
        if infer_row_pattern([label for label in row_labels if label]) in {"numeric runs", "F-numbered rows"}:
            structure_score += 2.0
        if "inline_formulation_table_recovered" in quality_flags:
            structure_score += 4.0
        if has_strong_formulation_table_signal(blob):
            structure_score += 2.0
        if any(token in blob for token in ["theoretical concentration", "final concentration", "formulation characters"]):
            structure_score += 3.0
        if "noise_rows_filtered" in quality_flags and cue_score + structure_score < 6.0:
            penalty_score -= 2.0
        if section_kind == "optimization" and cue_score < 3.0:
            penalty_score -= 2.0
    if role == "OPTIMIZATION_RESULT" and ("optimized" in blob or "desirability" in blob):
        structure_score += 2.0
    if role == "FORMULATION_TABLE" and meta.get("fraction_numeric_cells", 0) and float(meta.get("fraction_numeric_cells", 0)) < 0.03:
        penalty_score -= 2.0
    if role == "FORMULATION_TABLE" and (
        "figure_like_caption" in quality_flags
        or (preview_starts_figure and row_id_hits == 0)
        or (
            is_obvious_figure_or_front_matter_table(blob)
            and row_id_hits == 0
            and "table " not in blob.lower()
        )
    ):
        penalty_score -= 8.0
    if role == "VARIABLE_TABLE" and len(rows) < 6:
        penalty_score -= 1.0
    raw_score_component = raw_score / 25.0
    if role == "VARIABLE_TABLE" and cue_score + structure_score < 5.0:
        raw_score_component = min(raw_score_component, 1.0)
    if role == "FORMULATION_TABLE" and cue_score + structure_score < 4.0:
        raw_score_component = min(raw_score_component, 4.0)
    final_score = heading_score + cue_score + structure_score + penalty_score + raw_score_component
    return {
        "heading_score": heading_score,
        "cue_score": cue_score,
        "structure_score": structure_score,
        "penalty_score": penalty_score,
        "final_score": final_score,
    }


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


def choose_required_roles(signals: dict[str, bool]) -> tuple[str, str | None, list[str]]:
    if signals.get("has_doe_signal"):
        return GENERAL_SELECTOR_PROFILE, DOE_SELECTOR_OVERLAY, list(DOE_REQUIRED_ROLES)
    return GENERAL_SELECTOR_PROFILE, None, list(GENERAL_REQUIRED_ROLES)


def build_selector_surface_text(
    text_content: str,
    *,
    section_kind: str,
    role_candidate: str | None,
    split_trigger: str,
    table_role_hint: str | None = None,
) -> str:
    prefixes: list[str] = []
    if role_candidate == "PREPARATION_METHOD" or section_kind in {"preparation", "variant_preparation"}:
        prefixes.append("Preparation method:")
    if role_candidate == "MATERIALS" or section_kind == "materials":
        prefixes.append("Materials:")
    if section_kind == "experimental_design":
        prefixes.append("Experimental design: varying concentrations or batches.")
    if section_kind == "optimization":
        prefixes.append("Optimization result: best/highest/efficiency/loading outcome.")
    if table_role_hint == "design matrix":
        prefixes.append("Variable table: formulation factors and concentrations.")
    elif table_role_hint == "formulation":
        prefixes.append("Formulation table: drug/polymer ratio, loading, EE, DL.")
    elif table_role_hint == "optimization":
        prefixes.append("Optimization result table.")
    if split_trigger in {"table_inline_split", "post_table_split"} and "table " in text_content.lower():
        prefixes.append("Inline table/result split candidate.")
    if not prefixes:
        return text_content
    return " ".join(prefixes + [text_content]).strip()


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
        role_candidate = None
        method_cue_count = count_segmentation_cues(text_content, SEGMENT_METHOD_CUES)
        explicit_preparation_label = "preparation" in normalize_text(section_label).lower()
        classified_block_type, _, _ = classify_ordered_paragraph_block(text_content)
        if is_inline_table_candidate:
            role_candidate = "FORMULATION_TABLE"
        elif section_kind in {"preparation", "variant_preparation"} and (
            explicit_preparation_label
            or text_content.lower().startswith("empty nanospheres were prepared")
            or text_content.lower().startswith("empty nanocapsules were prepared")
            or (method_cue_count >= 2 and classified_block_type == "synthesis_method")
        ):
            block_type = "synthesis_method"
            role_candidate = "PREPARATION_METHOD"
        elif section_kind == "materials":
            block_type = "materials_procurement"
            role_candidate = "MATERIALS"
        candidate_id = f"{record['key']}__candidate_paragraph__{idx:02d}"
        origin_locator = f"{to_repo_rel(text_path)}#paragraph:{entry['paragraph_index']}#segment:{entry.get('segment_index', 0)}"
        selector_text_content = build_selector_surface_text(
            text_content,
            section_kind="table_related" if is_inline_table_candidate else section_kind,
            role_candidate=role_candidate,
            split_trigger="inline_table_recovery" if is_inline_table_candidate else split_trigger,
            table_role_hint=table_role_hint,
        )
        candidate_payload = {
            "candidate_id": candidate_id,
            "candidate_type": "table" if is_inline_table_candidate else "prose",
            "block_type": block_type,
            "role_candidate": role_candidate,
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
            "role_candidate": "FORMULATION_TABLE",
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
                    role_candidate="FORMULATION_TABLE",
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
            "noise_filtering": True,
        },
        "coverage_summary": {
            "total_candidates": len(candidates),
            "prose_candidates": sum(1 for item in candidates if item["candidate_type"] == "prose"),
            "table_candidates": sum(1 for item in candidates if item["candidate_type"] == "table"),
            "repaired_table_candidates": sum(1 for item in candidates if normalize_text(item.get("representation_status")) in {"repaired_summary", "repair_insufficient"}),
            "authoritative_stage1_table_repairs": sum(1 for item in candidates if normalize_text(item.get("repair_primary_source")) == "stage1_selected_table_asset"),
            "candidates_with_noise_flags": sum(1 for item in candidates if item.get("noise_flags")),
            "candidates_with_quality_flags": sum(1 for item in candidates if item.get("quality_flags")),
            "has_doe_signal": signals["has_doe_signal"],
            "has_sequential_signal": signals["has_sequential_signal"],
            "has_optimization_signal": signals["has_optimization_signal"],
        },
        "candidate_blocks": candidates,
    }
    return artifact, {"selector_candidates": selector_candidates, "signals": signals}


def build_role_aware_selection(
    *,
    segmented_candidates: list[dict[str, Any]],
    signals: dict[str, bool],
) -> dict[str, Any]:
    selector_profile, archetype_overlay, required_roles = choose_required_roles(signals)
    paragraph_roles = ["PREPARATION_METHOD", "MATERIALS", "EXPERIMENTAL_DESIGN", "FORMULATION_RESULT", "OPTIMIZATION_RESULT", "CONTEXT_FALLBACK"]
    table_roles = ["VARIABLE_TABLE", "FORMULATION_TABLE", "OPTIMIZATION_RESULT"]
    candidates_by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in set(required_roles + paragraph_roles + table_roles)}
    all_candidates: list[dict[str, Any]] = []
    paragraph_candidates = [item for item in segmented_candidates if item["candidate_kind"] == "paragraph"]
    total_paragraphs = max(1, len(paragraph_candidates))

    for candidate in paragraph_candidates:
        entry = {
            "text": candidate["text_content"],
            "paragraph_index": candidate.get("paragraph_index", 0),
            "segment_index": candidate.get("segment_index", 0),
        }
        for role in paragraph_roles:
            score = paragraph_role_score(entry, role, total_paragraphs=total_paragraphs)
            candidates_by_role.setdefault(role, []).append(
                {
                    "candidate_kind": candidate["candidate_kind"],
                    "candidate_id": candidate["candidate_id"],
                    "role": role,
                    "origin_key": candidate["origin_key"],
                    "source_type": candidate["source_type"],
                    "origin_locator": candidate["origin_locator"],
                    "entry": entry,
                    "score_breakdown": score,
                    "text_content": candidate["text_content"],
                    "duplicate_signature": "",
                    "section_label": candidate.get("section_label", ""),
                    "section_kind": candidate.get("section_kind", ""),
                    "noise_flags": list(candidate.get("noise_flags") or []),
                    "quality_flags": list(candidate.get("quality_flags") or []),
                }
            )
        formulation_table_score = paragraph_formulation_table_score(candidate)
        if float(formulation_table_score["final_score"]) > 0:
            candidates_by_role.setdefault("FORMULATION_TABLE", []).append(
                {
                    "candidate_kind": candidate["candidate_kind"],
                    "candidate_id": candidate["candidate_id"],
                    "role": "FORMULATION_TABLE",
                    "origin_key": candidate["origin_key"],
                    "source_type": candidate["source_type"],
                    "origin_locator": candidate["origin_locator"],
                    "entry": entry,
                    "score_breakdown": formulation_table_score,
                    "text_content": candidate["text_content"],
                    "duplicate_signature": "",
                    "section_label": candidate.get("section_label", ""),
                    "section_kind": candidate.get("section_kind", ""),
                    "noise_flags": list(candidate.get("noise_flags") or []),
                    "quality_flags": list(candidate.get("quality_flags") or []),
                }
            )

    for candidate in [item for item in segmented_candidates if item["candidate_kind"] == "table"]:
        item = candidate.get("item")
        signature = table_duplicate_signature(item) if isinstance(item, dict) else normalize_text(candidate.get("origin_locator"))
        for role in table_roles:
            score = table_role_score(candidate, role)
            candidates_by_role.setdefault(role, []).append(
                {
                    "candidate_kind": candidate["candidate_kind"],
                    "candidate_id": candidate["candidate_id"],
                    "role": role,
                    "origin_key": candidate["origin_key"],
                    "source_type": candidate["source_type"],
                    "origin_locator": candidate["origin_locator"],
                    "item": item,
                    "score_breakdown": score,
                    "text_content": candidate["text_content"],
                    "duplicate_signature": signature,
                    "section_label": candidate.get("section_label", ""),
                    "section_kind": candidate.get("section_kind", ""),
                    "noise_flags": list(candidate.get("noise_flags") or []),
                    "quality_flags": list(candidate.get("quality_flags") or []),
                }
            )

    for role, items in candidates_by_role.items():
        items.sort(
            key=lambda item: (
                -float(item["score_breakdown"]["final_score"]),
                item["origin_key"],
            )
        )
        all_candidates.extend(items)

    selected: list[dict[str, Any]] = []
    used_origins: set[str] = set()
    used_table_signatures: set[str] = set()
    missing_or_weak_roles: list[str] = []
    duplicate_suppression_events: list[dict[str, str]] = []
    selected_by_role: dict[str, list[dict[str, Any]]] = {}

    def maybe_add_candidate(candidate: dict[str, Any], priority: str) -> bool:
        origin_key = candidate["origin_key"]
        signature = candidate.get("duplicate_signature") or ""
        if origin_key in used_origins:
            return False
        if candidate["candidate_kind"] == "table" and signature and signature in used_table_signatures:
            duplicate_suppression_events.append(
                {
                    "role": candidate["role"],
                    "origin_locator": candidate["origin_locator"],
                    "reason": "duplicate_table_signature",
                }
            )
            return False
        used_origins.add(origin_key)
        if candidate["candidate_kind"] == "table" and signature:
            used_table_signatures.add(signature)
        chosen = dict(candidate)
        chosen["role_priority"] = priority
        selected.append(chosen)
        selected_by_role.setdefault(str(candidate["role"]), []).append(chosen)
        return True

    def remove_selected_candidate(candidate_id: str) -> None:
        nonlocal selected
        removed = [item for item in selected if normalize_text(item.get("candidate_id")) == candidate_id]
        if not removed:
            return
        selected = [item for item in selected if normalize_text(item.get("candidate_id")) != candidate_id]
        used_origins.clear()
        used_table_signatures.clear()
        selected_by_role.clear()
        for item in selected:
            used_origins.add(item["origin_key"])
            signature = item.get("duplicate_signature") or ""
            if item["candidate_kind"] == "table" and signature:
                used_table_signatures.add(signature)
            selected_by_role.setdefault(str(item["role"]), []).append(item)

    def materials_candidate_complete(candidate: dict[str, Any]) -> bool:
        return is_materials_inventory_candidate(candidate.get("text_content", ""))

    def preparation_candidate_complete(candidate: dict[str, Any]) -> bool:
        text = candidate.get("text_content", "")
        return has_preparation_procedure_signal(text) and not is_assay_comparator_dominated(text)

    def optimization_candidate_complete(candidate: dict[str, Any]) -> bool:
        return bool(optimization_decision_signature(candidate.get("text_content", "")))

    def add_best_materials_complement() -> bool:
        for candidate in candidates_by_role.get("MATERIALS", []):
            if materials_candidate_complete(candidate) and maybe_add_candidate(candidate, "coverage"):
                return True
        return False

    def correct_preparation_selection() -> bool:
        current = selected_by_role.get("PREPARATION_METHOD", [])
        current_primary = current[0] if current else None
        if current_primary and preparation_candidate_complete(current_primary):
            return False
        replacement = None
        for candidate in candidates_by_role.get("PREPARATION_METHOD", []):
            if preparation_candidate_complete(candidate):
                replacement = candidate
                break
        if replacement is None:
            return False
        if current_primary is not None:
            remove_selected_candidate(normalize_text(current_primary.get("candidate_id")))
        return maybe_add_candidate(replacement, "primary")

    def add_preparation_complement() -> bool:
        current = selected_by_role.get("PREPARATION_METHOD", [])
        if not current:
            return False
        combined_text = " ".join(item.get("text_content", "") for item in current)
        if all(token in normalize_text(combined_text).lower() for token in ["added", "evaporation", "centrifuged"]):
            return False
        for candidate in candidates_by_role.get("PREPARATION_METHOD", []):
            if normalize_text(candidate.get("candidate_id")) in {
                normalize_text(item.get("candidate_id")) for item in current
            }:
                continue
            text = candidate.get("text_content", "")
            if not preparation_candidate_complete(candidate):
                continue
            if not any(token in normalize_text(text).lower() for token in ["evaporation", "filtered", "centrifuged"]):
                continue
            if maybe_add_candidate(candidate, "coverage"):
                return True
        return False

    def add_optimization_complements() -> int:
        required_signatures = {"selected_condition", "optimal_formulation"}
        decision_signatures = {
            optimization_decision_signature(item.get("text_content", ""))
            for item in selected_by_role.get("OPTIMIZATION_RESULT", [])
            if optimization_decision_signature(item.get("text_content", "")) in required_signatures
        }
        added = 0
        for candidate in candidates_by_role.get("OPTIMIZATION_RESULT", []):
            signature = optimization_decision_signature(candidate.get("text_content", ""))
            if signature not in required_signatures or signature in decision_signatures:
                continue
            if maybe_add_candidate(candidate, "coverage"):
                decision_signatures.add(signature)
                added += 1
            if added >= 1:
                break
        return added

    for role in required_roles:
        role_candidates = candidates_by_role.get(role, [])
        threshold = ROLE_THRESHOLD_BY_ROLE.get(role, 1.0)
        chosen = None
        for candidate in role_candidates:
            if float(candidate["score_breakdown"]["final_score"]) < threshold:
                continue
            if maybe_add_candidate(candidate, "primary"):
                chosen = candidate
                break
        if chosen is None:
            missing_or_weak_roles.append(role)

    for role in SECONDARY_ELIGIBLE_ROLES:
        role_candidates = candidates_by_role.get(role, [])
        threshold = ROLE_THRESHOLD_BY_ROLE.get(role, 1.0) + SECONDARY_THRESHOLD_BONUS_BY_ROLE.get(role, 1.0)
        for candidate in role_candidates:
            if float(candidate["score_breakdown"]["final_score"]) < threshold:
                continue
            if maybe_add_candidate(candidate, "secondary"):
                break

    if "CONTEXT_FALLBACK" not in {item["role"] for item in selected}:
        for candidate in candidates_by_role.get("CONTEXT_FALLBACK", []):
            if maybe_add_candidate(candidate, "fallback"):
                break

    if "MATERIALS" in missing_or_weak_roles and add_best_materials_complement():
        missing_or_weak_roles = [role for role in missing_or_weak_roles if role != "MATERIALS"]

    correct_preparation_selection()
    add_preparation_complement()

    if signals.get("has_sequential_signal") or signals.get("has_optimization_signal"):
        add_optimization_complements()

    selected.sort(
        key=lambda item: (
            {"primary": 0, "secondary": 1, "coverage": 2, "fallback": 3}.get(item["role_priority"], 4),
            required_roles.index(item["role"]) if item["role"] in required_roles else 99,
            item["origin_key"],
        )
    )
    return {
        "selector_profile": selector_profile,
        "archetype_overlay": archetype_overlay,
        "required_roles": required_roles,
        "selected_roles": [item["role"] for item in selected],
        "missing_or_weak_roles": missing_or_weak_roles,
        "selected_candidates": selected,
        "duplicate_suppression_events": duplicate_suppression_events,
        "duplicate_table_suppression_active": bool(duplicate_suppression_events),
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
    current_table_mode = table_mode()
    summary_enhanced = summary_first_column_enhancement_enabled()
    current_input_packing_mode = input_packing_mode()
    raw_text = text_path.read_text(encoding="utf-8", errors="replace")
    summary_candidates = collect_summary_table_candidates(table_dir) if table_dir is not None and table_dir.exists() else []
    signals = dict(segmentation_bundle.get("signals") or detect_pre_llm_signals(raw_text, summary_candidates if current_table_mode == "summary" else []))
    role_selection = build_role_aware_selection(
        segmented_candidates=list(segmentation_bundle.get("selector_candidates") or []),
        signals=signals,
    )
    evidence_blocks: list[dict[str, Any]] = []
    order_tokens: list[str] = []

    def append_block(
        *,
        block_id: str,
        block_type: str,
        source_type: str,
        candidate_id: str | None,
        origin_locator: str,
        selection_reason: str,
        selection_feature: str,
        rank_score: int | None,
        text_content: str,
        is_synthesis: bool | None,
        is_table_derived: bool | None,
        is_candidate_critical: bool | None,
        role_assignment: str | None,
        role_priority: str | None,
        role_score_breakdown: dict[str, Any] | None,
    ) -> None:
        if not normalize_text(text_content):
            return
        evidence_blocks.append(
            {
                "block_id": block_id,
                "block_type": block_type,
                "source_type": source_type,
                "candidate_id": candidate_id,
                "origin_locator": origin_locator,
                "selection_reason": selection_reason,
                "selection_feature": selection_feature,
                "rank_score": rank_score,
                "order_index": len(evidence_blocks),
                "text_content": text_content,
                "is_synthesis": is_synthesis,
                "is_table_derived": is_table_derived,
                "is_candidate_critical": is_candidate_critical,
                "role_assignment": role_assignment,
                "role_priority": role_priority,
                "role_score_breakdown": role_score_breakdown,
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
        is_synthesis=False,
        is_table_derived=False,
        is_candidate_critical=None,
        role_assignment=None,
        role_priority="primary",
        role_score_breakdown=None,
    )

    if current_input_packing_mode == ORDERED_INPUT_PACKING_MODE:
        role_counts: dict[str, int] = {}
        for candidate in role_selection["selected_candidates"]:
            role = str(candidate["role"])
            role_counts[role] = role_counts.get(role, 0) + 1
            block_suffix = role_counts[role]
            block_type = {
                "PREPARATION_METHOD": "synthesis_method",
                "MATERIALS": "materials_procurement",
                "EXPERIMENTAL_DESIGN": "experimental_design",
                "VARIABLE_TABLE": "table",
                "FORMULATION_TABLE": "table",
                "FORMULATION_RESULT": "paragraph",
                "OPTIMIZATION_RESULT": "table" if candidate["candidate_kind"] == "table" else "paragraph",
                "CONTEXT_FALLBACK": "paragraph",
            }.get(role, "paragraph")
            prefix = {
                "PREPARATION_METHOD": "[SYNTHESIS_METHOD_BLOCK]\n",
                "MATERIALS": "[MATERIALS_PROCUREMENT_BLOCK]\n",
                "EXPERIMENTAL_DESIGN": "[EXPERIMENTAL_DESIGN_BLOCK]\n",
                "FORMULATION_RESULT": "[FORMULATION_RESULT_BLOCK]\n",
                "OPTIMIZATION_RESULT": "[OPTIMIZATION_RESULT_BLOCK]\n" if candidate["candidate_kind"] == "paragraph" else "",
                "CONTEXT_FALLBACK": "[PARAGRAPH_BLOCK]\n",
            }.get(role, "")
            append_block(
                block_id=f"{record['key']}__{normalize_token(role.lower())}__{block_suffix:02d}",
                block_type=block_type,
                source_type=str(candidate["source_type"]),
                candidate_id=str(candidate["candidate_id"]),
                origin_locator=str(candidate["origin_locator"]),
                selection_reason=f"role_constrained_{normalize_token(role.lower())}_{candidate['role_priority']}",
                selection_feature="role_aware_selector_v1",
                rank_score=int(round(float(candidate["score_breakdown"]["final_score"]))) if candidate["score_breakdown"]["final_score"] is not None else None,
                text_content=(prefix + candidate["text_content"]) if prefix else candidate["text_content"],
                is_synthesis=role == "PREPARATION_METHOD",
                is_table_derived=candidate["candidate_kind"] == "table",
                is_candidate_critical=candidate["role_priority"] == "primary",
                role_assignment=role,
                role_priority=str(candidate["role_priority"]),
                role_score_breakdown=candidate["score_breakdown"],
            )
    else:
        raw_prefix_text = raw_text[:max_chars] if max_chars > 0 else raw_text
        append_block(
            block_id=f"{record['key']}__raw_prefix__01",
            block_type="raw_prefix",
            source_type="clean_text",
            candidate_id=None,
            origin_locator=f"{to_repo_rel(text_path)}#chars:0:{len(raw_prefix_text)}",
            selection_reason="default_raw_prefix_fallback",
            selection_feature="max_text_chars_prefix",
            rank_score=None,
            text_content=raw_prefix_text,
            is_synthesis=None,
            is_table_derived=False,
            is_candidate_critical=None,
            role_assignment="CONTEXT_FALLBACK",
            role_priority="fallback",
            role_score_breakdown=None,
        )
        if current_table_mode == "summary":
            for idx, item in enumerate(summary_candidates[:4], start=1):
                append_block(
                    block_id=f"{record['key']}__table_summary__{idx:02d}",
                    block_type="table",
                    source_type="table_summary",
                    candidate_id=None,
                    origin_locator=to_repo_rel(item["path"]),
                    selection_reason="summary_mode_selected_table",
                    selection_feature="score_summary_table",
                    rank_score=int(item["score"]),
                    text_content=render_summary_table_block(item, enhancement_enabled=summary_enhanced),
                    is_synthesis=False,
                    is_table_derived=True,
                    is_candidate_critical=None,
                    role_assignment="FORMULATION_TABLE",
                    role_priority="fallback",
                    role_score_breakdown=table_role_score(item, "FORMULATION_TABLE"),
                )
        elif table_dir is not None and table_dir.exists():
            for idx, path in enumerate(sorted(table_dir.glob("*.csv"))[:4], start=1):
                append_block(
                    block_id=f"{record['key']}__table_excerpt__{idx:02d}",
                    block_type="table",
                    source_type="table_excerpt",
                    candidate_id=None,
                    origin_locator=to_repo_rel(path),
                    selection_reason="full_mode_first_four_sorted_csv",
                    selection_feature="sorted_csv_first_4",
                    rank_score=None,
                    text_content=render_full_table_block(path),
                    is_synthesis=False,
                    is_table_derived=True,
                    is_candidate_critical=None,
                    role_assignment="FORMULATION_TABLE",
                    role_priority="fallback",
                    role_score_breakdown=None,
                )

    feature_activation_snapshot = {
        "ordered_evidence_packing": current_input_packing_mode == ORDERED_INPUT_PACKING_MODE,
        "summary_table_mode": current_table_mode == "summary",
        "table_selection_scoring": current_table_mode == "summary" and bool(summary_candidates),
        "selector_debug_available": current_table_mode == "summary" and table_dir is not None and table_dir.exists(),
        "candidate_segmentation_profile": SEGMENTATION_PROFILE,
        "section_aware_candidate_split": True,
        "candidate_table_isolation": table_dir is not None and table_dir.exists(),
        "candidate_noise_filtering": True,
        "doe_pre_llm_detection": signals["has_doe_signal"],
        "sequential_optimization_detection": signals["has_sequential_signal"],
        "role_aware_evidence_selection": current_input_packing_mode == ORDERED_INPUT_PACKING_MODE,
        "doe_overlay_selection": role_selection["archetype_overlay"] == DOE_SELECTOR_OVERLAY,
        "duplicate_table_suppression": role_selection["duplicate_table_suppression_active"],
    }
    coverage_summary = {
        "total_blocks": len(evidence_blocks),
        "synthesis_blocks": sum(1 for block in evidence_blocks if block["block_type"] == "synthesis_method"),
        "table_blocks": sum(1 for block in evidence_blocks if block["block_type"] == "table"),
        "caption_blocks": sum(1 for block in evidence_blocks if block["block_type"] == "caption"),
        "paragraph_blocks": sum(1 for block in evidence_blocks if block["block_type"] in {"paragraph", "raw_prefix", "experimental_design", "materials_procurement"}),
        "has_optimization_signal": signals["has_optimization_signal"],
        "has_doe_signal": signals["has_doe_signal"],
        "has_sequential_signal": signals["has_sequential_signal"],
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
                "text_content",
                "role_assignment",
                "role_priority",
                "role_score_breakdown",
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
    input_contract_satisfied = (
        current_input_packing_mode == ORDERED_INPUT_PACKING_MODE
        and current_table_mode == "summary"
        and feature_activation_snapshot["table_selection_scoring"]
        and feature_activation_snapshot["selector_debug_available"]
        and feature_activation_snapshot["role_aware_evidence_selection"]
    )
    required_roles_present = sorted({role for role in role_selection["required_roles"] if role in role_selection["selected_roles"]})
    required_features_active = input_contract_satisfied and not role_selection["missing_or_weak_roles"]
    design_status = {
        "input_contract_satisfied": input_contract_satisfied,
        "required_features_active": required_features_active,
        "required_roles_present": required_roles_present,
        "missing_or_weak_roles": list(role_selection["missing_or_weak_roles"]),
        "overall": "pass" if input_contract_satisfied and required_features_active else "fail",
        "nonconformance_reasons": [
            reason
            for reason, active in [
                ("ordered_evidence_packing_inactive", feature_activation_snapshot["ordered_evidence_packing"]),
                ("summary_table_mode_inactive", feature_activation_snapshot["summary_table_mode"]),
                ("table_selection_scoring_inactive", feature_activation_snapshot["table_selection_scoring"]),
                ("selector_debug_unavailable", feature_activation_snapshot["selector_debug_available"]),
                ("role_aware_selector_inactive", feature_activation_snapshot["role_aware_evidence_selection"]),
            ]
            if not active
        ]
        + [f"missing_or_weak_role:{role}" for role in role_selection["missing_or_weak_roles"]],
    }
    ordered_block_order = order_tokens if order_tokens else ["none"]
    artifact = {
        "paper_key": record["key"],
        "source_clean_text_path": to_repo_rel(text_path),
        "source_manifest_path": to_repo_rel(manifest_path),
        "source_scope_manifest_path": to_repo_rel(manifest_path),
        "source_candidate_artifact_path": to_repo_rel(candidate_artifact_path),
        "input_contract": {
            "input_packing_mode": current_input_packing_mode,
            "table_mode": current_table_mode,
            "summary_first_column_enhancement": summary_enhanced,
            "ordered_block_order": ordered_block_order,
            "selector_strategy": "role_aware_selector_v1" if current_input_packing_mode == ORDERED_INPUT_PACKING_MODE else resolved_selector_strategy(current_table_mode),
        },
        "producer_script": producer_script,
        "contract_version": "s2_2_evidence_blocks_v1",
        "segmentation_profile": SEGMENTATION_PROFILE,
        "selector_profile": role_selection["selector_profile"] if current_input_packing_mode == ORDERED_INPUT_PACKING_MODE else None,
        "archetype_overlay": role_selection["archetype_overlay"] if current_input_packing_mode == ORDERED_INPUT_PACKING_MODE else None,
        "evidence_blocks": evidence_blocks,
        "coverage_summary": coverage_summary,
        "required_roles": role_selection["required_roles"] if current_input_packing_mode == ORDERED_INPUT_PACKING_MODE else [],
        "selected_roles": role_selection["selected_roles"] if current_input_packing_mode == ORDERED_INPUT_PACKING_MODE else [],
        "missing_or_weak_roles": role_selection["missing_or_weak_roles"] if current_input_packing_mode == ORDERED_INPUT_PACKING_MODE else [],
        "duplicate_table_suppression_events": role_selection["duplicate_suppression_events"] if current_input_packing_mode == ORDERED_INPUT_PACKING_MODE else [],
        "feature_activation_snapshot": feature_activation_snapshot,
        "technical_status": technical_status,
        "design_status": design_status,
    }
    debug_payload = None
    if current_table_mode == "summary" and table_dir is not None and table_dir.exists():
        selected_summary_names = {
            Path(candidate["origin_locator"]).name
            for candidate in role_selection["selected_candidates"]
            if candidate["candidate_kind"] == "table"
        }
        payload_candidates: list[dict[str, Any]] = []
        for item in summary_candidates:
            rows = item["rows"]
            first_data_row = rows[1] if len(rows) > 1 else rows[0]
            selected_candidate = next(
                (
                    candidate
                    for candidate in role_selection["selected_candidates"]
                    if candidate["candidate_kind"] == "table" and candidate["origin_locator"] == to_repo_rel(item["path"])
                ),
                None,
            )
            payload_candidates.append(
                {
                    "file": item["path"].name,
                    "score": item["score"],
                    "selected": item["path"].name in selected_summary_names,
                    "selected_role": selected_candidate["role"] if selected_candidate is not None else "",
                    "selected_role_priority": selected_candidate["role_priority"] if selected_candidate is not None else "",
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
        debug_payload = {
            "document_key": record["key"],
            "table_mode": current_table_mode,
            "selection_ranking_mode": "role_aware_selector_v1" if current_input_packing_mode == ORDERED_INPUT_PACKING_MODE else resolved_selector_strategy(current_table_mode),
            "summary_first_column_enhancement": "yes" if summary_enhanced else "no",
            "candidate_artifact_path": to_repo_rel(candidate_artifact_path),
            "max_tables": 4,
            "selector_profile": artifact.get("selector_profile") or "",
            "archetype_overlay": artifact.get("archetype_overlay") or "",
            "required_roles": artifact.get("required_roles") or [],
            "missing_or_weak_roles": artifact.get("missing_or_weak_roles") or [],
            "selected_tables": sorted(selected_summary_names),
            "duplicate_table_suppression_events": artifact.get("duplicate_table_suppression_events") or [],
            "candidates": payload_candidates,
        }
    return artifact, debug_payload


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
    paper_text_index = prompt_text.find("Paper text:\n")
    evidence_pack_index = prompt_text.find("Evidence pack:\n")
    table_excerpts_index = prompt_text.find("Table excerpts:\n")
    evidence_blocks = [
        block
        for block in ensure_list(evidence_artifact.get("evidence_blocks"))
        if isinstance(block, dict) and normalize_text(block.get("text_content"))
    ]
    normalized_prompt_text = normalize_text(prompt_text)
    exact_payload_counts = Counter(normalize_text(block.get("text_content")) for block in evidence_blocks)
    exact_duplicate_block_count = sum(count - 1 for count in exact_payload_counts.values() if count > 1)
    all_selected_blocks_included = all(
        normalize_text(block.get("text_content")) in normalized_prompt_text for block in evidence_blocks
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
    return {
        "document_key": document["document_key"],
        "doi": document["doi"],
        "source_mode": document["source_mode"],
        "table_mode": table_mode_value,
        "summary_first_column_enhancement": "yes" if summary_enhanced else "no",
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
        "prompt_length": len(prompt_text),
        "prompt_head_preview": prompt_head_preview,
        "prompt_tail_preview": prompt_tail_preview,
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
                "unresolved_reason": normalize_text(candidate.get("unresolved_reason")),
                "raw_table_preview": normalize_text(candidate.get("raw_table_preview"))[:260],
                "repaired_table_preview": normalize_text(candidate.get("repaired_table_preview"))[:260],
                "text_preview": normalize_text(candidate.get("text_preview") or candidate.get("text_content"))[:260],
            }
        )
    return rows


def build_live_prompt(record: dict[str, str], evidence_artifact: dict[str, Any]) -> str:
    input_contract = evidence_artifact.get("input_contract", {})
    current_table_mode = normalize_text(input_contract.get("table_mode")).lower() or table_mode()
    summary_enhanced = bool(input_contract.get("summary_first_column_enhancement"))
    current_input_packing_mode = normalize_text(input_contract.get("input_packing_mode")).lower() or input_packing_mode()
    ordered_block_order = [str(value) for value in ensure_list(input_contract.get("ordered_block_order")) if str(value).strip()]
    ordered_input_enabled = current_input_packing_mode == ORDERED_INPUT_PACKING_MODE
    ordered_blocks = [
        block.get("text_content", "")
        for block in ensure_list(evidence_artifact.get("evidence_blocks"))
        if isinstance(block, dict) and normalize_text(block.get("text_content"))
    ]
    paper_text = "\n\n".join(
        block.get("text_content", "")
        for block in ensure_list(evidence_artifact.get("evidence_blocks"))
        if isinstance(block, dict)
        and normalize_text(block.get("block_type")) not in {"metadata", "table"}
        and normalize_text(block.get("text_content"))
    ).strip()
    table_text = "\n\n".join(
        block.get("text_content", "")
        for block in ensure_list(evidence_artifact.get("evidence_blocks"))
        if isinstance(block, dict)
        and normalize_text(block.get("block_type")) == "table"
        and normalize_text(block.get("text_content"))
    ).strip()
    schema = {
        "document_key": record["key"],
        "doi": record["doi"],
        "formulation_candidates": [
            {
                "candidate_id": "string",
                "raw_label": "string",
                "normalized_label": "string",
                "instance_kind": "new_formulation|variant_formulation|candidate_non_formulation|unclear",
                "formulation_role": "variant|baseline|optimized|control|unclear",
                "parent_candidate_id": "string or empty",
                "ambiguity_note": "string or empty",
                "evidence_span_ids": ["span ids"],
                "status": "reported|ambiguous|derived_from_shared_context",
            }
        ],
        "component_candidates": [
            {
                "component_id": "string",
                "formulation_candidate_id": "string",
                "component_name": "string",
                "component_role": "polymer|drug|surfactant|solvent|additive|unknown",
                "amount_text": "string or empty",
                "amount_kind": "concentration|mass|ratio|unknown",
                "phase_hint": "string or empty",
                "ambiguity_note": "string or empty",
                "evidence_span_ids": ["span ids"],
            }
        ],
        "variable_candidates": [
            {
                "variable_id": "string",
                "formulation_candidate_id": "string or empty for shared",
                "variable_name": "string",
                "value_text": "string",
                "variable_role": "identity_signal|process_setting|doe_factor|shared_context|unclear",
                "ambiguity_note": "string or empty",
                "evidence_span_ids": ["span ids"],
            }
        ],
        "measurement_candidates": [
            {
                "measurement_id": "string",
                "formulation_candidate_id": "string",
                "measurement_name": "size|pdi|zeta_potential|encapsulation_efficiency|loading_content|other",
                "value_text": "string",
                "unit_text": "string or empty",
                "ambiguity_note": "string or empty",
                "evidence_span_ids": ["span ids"],
            }
        ],
        "relation_hints": [
            {
                "relation_id": "string",
                "source_candidate_id": "string",
                "target_candidate_id": "string",
                "relation_type": "inherits_from|shares_context_with|varies_by|other",
                "note": "string",
                "evidence_span_ids": ["span ids"],
            }
        ],
        "evidence_spans": [
            {
                "span_id": "string",
                "source_region_type": "text_span|table_row|table_cell|table_caption|methods_sentence|paper_notes",
                "source_locator_text": "string",
                "supporting_text": "string",
            }
        ],
        "unassigned_observations": [
            {
                "observation_id": "string",
                "category": "reported_but_unassigned|measurement_without_boundary|shared_context|other",
                "note": "string",
                "evidence_span_ids": ["span ids"],
            }
        ],
        "table_formulation_scopes": [
            {
                "table_id": "string",
                "is_formulation_table": True,
                "table_type": "full_formulation|partial_formulation|sequential_child|doe_table|non_formulation",
                "confidence": "high|medium|low",
                "evidence_span": "string",
            }
        ],
        "table_variable_roles": [
            {
                "table_id": "string",
                "varying_variables": ["variable names"],
                "constant_variables": ["variable names"],
                "new_variables_introduced": ["variable names"],
            }
        ],
        "selection_markers": [
            {
                "marker_readiness": "execution_ready|partial_semantic",
                "source_table_id": "string",
                "selected_variable": "string",
                "selected_value": "string",
                "explicit": True,
                "evidence_span": "string",
            }
        ],
        "inheritance_markers": [
            {
                "marker_readiness": "execution_ready|partial_semantic",
                "from_table": "string",
                "to_table": "string",
                "inherit_type": "selected_condition",
                "variable": "string",
                "value": "string",
                "evidence_span": "string",
            }
        ],
        "preparation_inheritance_markers": [
            {
                "table_id": "string",
                "inherits_from_preparation": True,
                "evidence_span": "string",
            }
        ],
        "boundary_markers": [
            {
                "table_id": "string",
                "is_doe": False,
            }
        ],
    }
    table_mode_note = ""
    if current_table_mode == "summary":
        table_mode_note = (
            "- Table excerpts are provided in summary mode only.\n"
            + (
                "- Summary selection prioritizes DOE-like tables with explicit numbered row anchors.\n"
                if summary_enhanced
                else ""
            )
            + (
                "- First-column row labels / numbering previews are exposed for explicit row-anchor tables.\n"
                if summary_enhanced
                else ""
            )
            + "- Use them to understand table semantics, variable structure, header units, and row-identifier patterns.\n"
            + "- Do not infer that unsampled rows are fully enumerated in the prompt.\n"
            + "- Do not expand every table row into separate objects unless the paper text itself clearly defines stable formulation boundaries.\n"
        )
    controlled_input_note = ""
    if ordered_input_enabled:
        resolved_order_note = (
            f"- Resolved evidence block order: {' > '.join(ordered_block_order)}.\n"
            if ordered_block_order
            else ""
        )
        controlled_input_note = (
            "- Controlled evidence packing is enabled.\n"
            "- Prompt order prioritizes synthesis/preparation blocks, then materials/procurement blocks, then table evidence, then narrative fallback.\n"
            + resolved_order_note
            + "- Treat the evidence pack as the governed live input ordering for this run.\n"
        )
    if ordered_input_enabled:
        evidence_text = "\n\n".join(ordered_blocks).strip()
        input_block = "Evidence pack:\n" + f"{evidence_text}\n"
    else:
        input_block = "Paper text:\n" + f"{paper_text}\n\nTable excerpts:\n{table_text}\n"
    return (
        "You are extracting Stage2 v2 semantic objects for a governed comparator slice.\n"
        "Rules:\n"
        "- Preserve ambiguity explicitly.\n"
        "- Emit object-first outputs only.\n"
        "- Do not perform relation resolution, inheritance closure, or final-row materialization.\n"
        "- Do not force DOE rows if the paper only reports factors but not clear formulation boundaries.\n"
        "- Do not enumerate table rows.\n"
        "- Every top-level key in the schema is required; if a family is absent, return an empty list for that family.\n"
        "- Emit table-level markers when a table is formulation-bearing, including sweep tables whose rows represent explicit formulation variants under fixed context.\n"
        "- Use literal paper table labels such as 'Table 1' or 'Table 2' for every marker table_id field; do not use file names or asset names.\n"
        "- Prefer true paper caption lines and nearby narrative references over noisy PDF extractor fragments when deciding which paper table a marker belongs to.\n"
        "- Do not mark a table as non_formulation just because some extracted PDF table fragments contain abstract text, references, or assay traces; use the paper's caption and surrounding formulation narrative to identify the real formulation tables.\n"
        "- Mark a later sweep table as sequential_child when it is performed under a selected condition from an earlier table.\n"
        "- Emit selection_markers and inheritance_markers only from explicit text or explicit table notes/captions; do not guess beyond the paper evidence.\n"
        f"- Use marker_readiness='{EXECUTION_READY_MARKER}' only when the marker is fully grounded for current execution use.\n"
        f"- Use marker_readiness='{PARTIAL_SEMANTIC_MARKER}' when the paper clearly supports the semantic cue but some non-execution-critical grounding is still incomplete.\n"
        "- For selection_markers, source_table_id, selected_variable, and selected_value may remain empty only when marker_readiness is partial_semantic, except for the explicit sequential-optimization literal-value pattern described below.\n"
        "- For inheritance_markers, from_table and to_table may remain empty only when marker_readiness is partial_semantic, except for the explicit sequential-optimization literal-value pattern described below.\n"
        "- For inheritance_markers, inherit_type, variable, and value remain strict and must stay concrete whenever you emit the marker.\n"
        "- When the paper gives a concrete selected value such as '3 mg/mL' or '10:1', selected_value must be that literal value, not a placeholder such as 'optimal concentration' or 'optimal ratio'.\n"
        "- Mandatory sequential-optimization literal-value rule: when a variable is explicitly explored over multiple concrete values, the paper explicitly states that one concrete value was selected or chosen as optimal, and nearby later text explicitly says that this chosen optimal setting was reused or carried forward in following experiments, you MUST extract that literal chosen value and keep it literal.\n"
        "- For that rule, you MUST emit an execution_ready selection_marker with the literal selected_value and an execution_ready inheritance_marker with inherit_type='selected_condition' and the same literal value.\n"
        "- For that rule, the selection sentence and the reuse sentence may appear in the same paragraph, the adjacent paragraph, or the nearby discussion immediately following the relevant optimization table.\n"
        "- For that rule, the optimal value MUST be explicitly stated as a concrete literal such as '3 mg/mL' or '10:1', and reuse MUST be explicitly stated with wording such as 'selected', 'chosen', 'used for the remaining studies', or 'after ... had been determined'.\n"
        "- For that explicit sequential-optimization literal-value rule only, selection_markers may be execution_ready with empty source_table_id if selected_variable and literal selected_value are explicit, and inheritance_markers may be execution_ready with empty from_table and to_table if inherit_type, variable, and literal value are explicit.\n"
        "- Anti-placeholder rule: you MUST NOT output abstract placeholders such as 'optimal concentration', 'optimal ratio', or 'optimal formulation' when a concrete literal value is explicitly stated in that nearby local evidence window. If a nearby explicit literal value exists, using the abstract placeholder instead of that literal is incorrect.\n"
        "- If no explicit literal value is present in that nearby local evidence window, abstract wording alone must stay partial_semantic and must NOT become execution_ready.\n"
        "- Do not guess values and do not infer them from unrelated tables, distant sections, or full-document fallback.\n"
        "- For partial markers, keep every grounded field that the paper supports and leave only the still-unresolved non-critical fields empty.\n"
        "- Do not use vague placeholders to simulate grounding.\n"
        "- Emit preparation_inheritance_markers only when the paper makes clear that a formulation-bearing table inherits the shared preparation context.\n"
        "- boundary_markers must explicitly label each marked table as DOE or non-DOE.\n"
        "- Example sequential-optimization pattern: if Table 1 varies one formulation variable across explicit levels and the paper says one level was selected as optimal, and Table 2 then varies a different formulation variable under that selected condition, emit both Table 1 and Table 2 in table_formulation_scopes, emit the Table 1 selection in selection_markers, and emit a Table 1 -> Table 2 selected_condition inheritance marker.\n"
        "- Minimal example for that pattern:\n"
        "  table_formulation_scopes = [{\"table_id\": \"Table 1\", \"is_formulation_table\": true, \"table_type\": \"partial_formulation\", \"confidence\": \"high\", \"evidence_span\": \"...\"}, {\"table_id\": \"Table 2\", \"is_formulation_table\": true, \"table_type\": \"sequential_child\", \"confidence\": \"high\", \"evidence_span\": \"...\"}]\n"
        f"  selection_markers = [{{\"marker_readiness\": \"{EXECUTION_READY_MARKER}\", \"source_table_id\": \"Table 1\", \"selected_variable\": \"poloxamer 188 concentration\", \"selected_value\": \"3 mg/mL\", \"explicit\": true, \"evidence_span\": \"...selected as optimal...\"}}]\n"
        f"  inheritance_markers = [{{\"marker_readiness\": \"{EXECUTION_READY_MARKER}\", \"from_table\": \"Table 1\", \"to_table\": \"Table 2\", \"inherit_type\": \"selected_condition\", \"variable\": \"poloxamer 188 concentration\", \"value\": \"3 mg/mL\", \"evidence_span\": \"...after the optimal surfactant concentration had been determined...\"}}]\n"
        "- Positive example for the mandatory sequential-optimization literal-value rule:\n"
        "  nearby evidence = 'Poloxamer 188 concentration was studied at 2.5, 3, 4, and 10 mg/mL. 3 mg/mL was selected as the optimal surfactant concentration. After the optimal surfactant concentration had been determined, the remaining studies used that condition.'\n"
        f"  selection_markers = [{{\"marker_readiness\": \"{EXECUTION_READY_MARKER}\", \"source_table_id\": \"\", \"selected_variable\": \"poloxamer 188 concentration\", \"selected_value\": \"3 mg/mL\", \"explicit\": true, \"evidence_span\": \"3 mg/mL was selected as the optimal surfactant concentration.\"}}]\n"
        f"  inheritance_markers = [{{\"marker_readiness\": \"{EXECUTION_READY_MARKER}\", \"from_table\": \"\", \"to_table\": \"\", \"inherit_type\": \"selected_condition\", \"variable\": \"poloxamer 188 concentration\", \"value\": \"3 mg/mL\", \"evidence_span\": \"After the optimal surfactant concentration had been determined, the remaining studies used that condition.\"}}]\n"
        "- Negative example for the same rule:\n"
        "  nearby evidence = 'After the optimal surfactant concentration had been determined, the remaining studies used that condition.'\n"
        f"  selection_markers = [{{\"marker_readiness\": \"{PARTIAL_SEMANTIC_MARKER}\", \"source_table_id\": \"\", \"selected_variable\": \"surfactant concentration\", \"selected_value\": \"\", \"explicit\": true, \"evidence_span\": \"After the optimal surfactant concentration had been determined, the remaining studies used that condition.\"}}]\n"
        f"  inheritance_markers = [{{\"marker_readiness\": \"{PARTIAL_SEMANTIC_MARKER}\", \"from_table\": \"\", \"to_table\": \"\", \"inherit_type\": \"selected_condition\", \"variable\": \"surfactant concentration\", \"value\": \"\", \"evidence_span\": \"After the optimal surfactant concentration had been determined, the remaining studies used that condition.\"}}]\n"
        "- Partial example when the semantic cue is explicit but grounding is incomplete:\n"
        f"  selection_markers = [{{\"marker_readiness\": \"{PARTIAL_SEMANTIC_MARKER}\", \"source_table_id\": \"\", \"selected_variable\": \"surfactant concentration\", \"selected_value\": \"\", \"explicit\": true, \"evidence_span\": \"...the optimal surfactant concentration was then used...\"}}]\n"
        f"  inheritance_markers = [{{\"marker_readiness\": \"{PARTIAL_SEMANTIC_MARKER}\", \"from_table\": \"\", \"to_table\": \"\", \"inherit_type\": \"selected_condition\", \"variable\": \"surfactant concentration\", \"value\": \"3 mg/mL\", \"evidence_span\": \"...after the optimal surfactant concentration had been determined...\"}}]\n"
        "- Return valid JSON only.\n\n"
        f"Table mode: {current_table_mode}\n"
        f"{table_mode_note}\n"
        f"{controlled_input_note}\n"
        f"Paper key: {record['key']}\n"
        f"DOI: {record['doi']}\n"
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
) -> dict[str, Any]:
    sanitized_text, sanitization_audit = sanitize_stage2_json_text(raw_response_text)
    if json_sanitization_applied(sanitization_audit):
        write_json_sanitization_audit(raw_response_path, sanitization_audit)
    parsed = json.loads(sanitized_text)
    if is_live_v2_raw_response_shape(parsed):
        return normalize_replayed_live_document(record, parsed, raw_response_path)
    formulations = parsed.get("formulations") or []
    text_path = Path(record["text_path"])
    if not text_path.is_absolute():
        text_path = (PROJECT_ROOT / text_path).resolve()
    table_dir = resolve_tables_dir(text_path, record["key"])
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
        }
    )


def normalize_live_document(record: dict[str, str], parsed: dict[str, Any], raw_response_path: Path) -> dict[str, Any]:
    return build_live_v2_document(
        record=record,
        parsed=parsed,
        raw_response_path=raw_response_path,
        source_mode="live_llm_stage2_v2",
        replay_mode="none",
    )


def normalize_replayed_live_document(record: dict[str, str], parsed: dict[str, Any], raw_response_path: Path) -> dict[str, Any]:
    return build_live_v2_document(
        record=record,
        parsed=parsed,
        raw_response_path=raw_response_path,
        source_mode="saved_raw_live_v2_replay_to_stage2_v2",
        replay_mode="saved_raw_response_replay",
    )


def build_live_v2_document(
    *,
    record: dict[str, str],
    parsed: dict[str, Any],
    raw_response_path: Path,
    source_mode: str,
    replay_mode: str,
) -> dict[str, Any]:
    parsed = prune_invalid_live_inheritance_markers(parsed, raw_response_path)
    text_path = Path(record["text_path"])
    if not text_path.is_absolute():
        text_path = (PROJECT_ROOT / text_path).resolve()
    table_dir = resolve_tables_dir(text_path, record["key"])
    source_table_files = []
    if table_dir and table_dir.exists():
        source_table_files = [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in sorted(table_dir.glob("*.csv"))]
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
        }
    )


def infer_semantic_scope_declarations(document: dict[str, Any]) -> list[dict[str, Any]]:
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


def finalize_llm_first_document(document: dict[str, Any]) -> dict[str, Any]:
    augment_document_with_table_markers(document)
    document["stage2_semantic_source_mode"] = STAGE2_SEMANTIC_SOURCE_MODE
    document["semantic_universe_authority"] = LLM_SEMANTIC_DISCOVERY
    declarations = infer_semantic_scope_declarations(document)
    document["semantic_scope_declarations"] = declarations
    for item in document.get("formulation_candidates", []):
        if not isinstance(item, dict):
            continue
        candidate_id = normalize_text(item.get("candidate_id"))
        item["stage2_semantic_source_mode"] = normalize_text(item.get("stage2_semantic_source_mode")) or STAGE2_SEMANTIC_SOURCE_MODE
        item["semantic_universe_authority"] = normalize_text(item.get("semantic_universe_authority")) or LLM_SEMANTIC_DISCOVERY
        item["row_materialization_mode"] = normalize_text(item.get("row_materialization_mode")) or LLM_SEMANTIC_DISCOVERY
        item["semantic_scope_authority"] = normalize_text(item.get("semantic_scope_authority")) or LLM_DECLARED_SCOPE
        item["semantic_scope_ref"] = normalize_text(item.get("semantic_scope_ref")) or default_llm_scope_ref(document, candidate_id)
    return document


def summary_row(document: dict[str, Any]) -> dict[str, Any]:
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
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-sec", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_models_or_raise([args.model], context="stage2_objects_v2 extractor model check")

    manifest_path = Path(args.manifest_tsv)
    if not manifest_path.is_absolute():
        manifest_path = (PROJECT_ROOT / manifest_path).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    records = read_tsv(manifest_path)
    selected_keys = [normalize_text(key) for key in args.paper_keys if normalize_text(key)]
    if selected_keys:
        records = [record for record in records if normalize_text(record.get("key")) in selected_keys]
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
    table_selection_debug_rows: list[dict[str, Any]] = []
    candidate_segmentation_debug_rows: list[dict[str, Any]] = []
    prompt_preview_path = out_dir.parent / "analysis" / PROMPT_PREVIEW_NAME
    table_selection_debug_path = out_dir.parent / "analysis" / TABLE_SELECTION_DEBUG_NAME
    candidate_segmentation_debug_path = out_dir.parent / "analysis" / CANDIDATE_SEGMENTATION_DEBUG_NAME
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
                key = normalize_text(record.get("key"))
                if not key:
                    continue
                if "text_path" not in record or not normalize_text(record.get("text_path")):
                    raise ValueError(f"Manifest row for {key} is missing text_path.")

                progress_label = progress.begin_task(index, key)
                try:
                    text_path = Path(record["text_path"])
                    if not text_path.is_absolute():
                        text_path = (PROJECT_ROOT / text_path).resolve()
                    if not text_path.exists():
                        raise FileNotFoundError(f"Missing paper text for {key}: {text_path}")
                    table_dir = resolve_tables_dir(text_path, key)
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
                    prompt = build_live_prompt(record, evidence_artifact)
                    prompt_preview_rows.append(
                        build_prompt_preview_row(
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
                    )
                    if debug_payload is not None:
                        debug_payload["evidence_artifact_path"] = to_repo_rel(artifact_path)
                        table_selection_debug_rows.append(debug_payload)
                    if args.source_mode == "legacy_llm_replay":
                        assert legacy_raw_dir is not None
                        legacy_raw_path = find_legacy_raw_response(legacy_raw_dir, key)
                        raw_copy_path = raw_dir / legacy_raw_path.name
                        shutil.copy2(legacy_raw_path, raw_copy_path)
                        document = convert_legacy_raw_response_to_v2(
                            record=record,
                            raw_response_path=raw_copy_path,
                            raw_response_text=raw_copy_path.read_text(encoding="utf-8", errors="replace"),
                        )
                    else:
                        if args.llm_backend == "gemini":
                            raw_text = call_gemini(
                                args.model,
                                prompt,
                                args.request_retries,
                                args.retry_sleep_sec,
                                progress_label=progress_label,
                            )
                        else:
                            raw_text = call_nvidia_hosted(
                                args.model,
                                prompt,
                                args.request_retries,
                                args.retry_sleep_sec,
                                progress_label=progress_label,
                        )
                        raw_copy_path = raw_dir / f"{key}__stage2_v2_raw_response.json"
                        raw_copy_path.write_text(raw_text, encoding="utf-8")
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
                        document = normalize_live_document(record, parsed, raw_copy_path)

                    handle.write(json.dumps(document, ensure_ascii=False) + "\n")
                    summary_rows.append(summary_row(document))
                    progress.complete_task()
                except Exception as exc:
                    progress.fail_task(exc)
                    raise
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
                "prompt_length",
                "prompt_head_preview",
                "prompt_tail_preview",
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
                "unresolved_reason",
                "raw_table_preview",
                "repaired_table_preview",
                "text_preview",
            ],
        )
    print(f"wrote_jsonl={jsonl_path}")
    print(f"wrote_summary={summary_path}")
    print(f"wrote_candidate_blocks_dir={out_dir / CANDIDATE_BLOCKS_SUBDIR}")
    print(f"wrote_evidence_blocks_dir={out_dir / EVIDENCE_BLOCKS_SUBDIR}")
    if prompt_preview_rows:
        print(f"wrote_prompt_preview={prompt_preview_path}")
    if table_selection_debug_rows:
        print(f"wrote_table_selection_debug={table_selection_debug_path}")
    if candidate_segmentation_debug_rows:
        print(f"wrote_candidate_segmentation_debug={candidate_segmentation_debug_path}")
    print(f"wrote_raw_responses_dir={raw_dir}")


if __name__ == "__main__":
    main()
