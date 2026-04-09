#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
PROMPT_PREVIEW_NAME = "stage2_prompt_preview_v1.tsv"
TABLE_SELECTION_DEBUG_NAME = "table_selection_debug_v1.json"
MARKER_CLEANUP_AUDIT_SUFFIX = "__marker_cleanup_audit.json"
NVIDIA_HOSTED_CHAT_COMPLETIONS_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LEGACY_FIELD_ALIASES = {"plga_mw_kDa": "polymer_mw_kDa"}
SUMMARY_FIRST_COLUMN_ENHANCEMENT_ENV = "STAGE2_TABLE_SUMMARY_FIRST_COLUMN_ENHANCEMENT"
INPUT_PACKING_MODE_ENV = "STAGE2_INPUT_EVIDENCE_PACKING_MODE"
ORDERED_INPUT_PACKING_MODE = "ordered_blocks"
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


def score_summary_table(path: Path, rows: list[list[str]], meta: dict[str, Any]) -> tuple[int, str]:
    header_row = rows[0] if rows else []
    data_rows = rows[1:] if len(rows) > 1 else []
    header_parts: list[str] = []
    for cell in header_row:
        header_parts.extend(parse_header_cell(cell))
    header_parts = [part for part in header_parts if part]
    row_ids = [row[0] for row in data_rows if row and normalize_text(row[0])]
    row_pattern = infer_row_pattern(row_ids)
    role_hint = infer_table_role_hint(header_parts, meta)
    numeric_rows = sum(1 for value in row_ids if parse_numeric_row_label(value) is not None)
    numeric_ratio = numeric_rows / max(len(data_rows), 1)
    signal_text = build_summary_table_signal_text(rows, meta, header_parts)
    score = 0
    if row_pattern in {"numeric runs", "F-numbered rows"}:
        score += 100
    if role_hint == "design matrix":
        score += 80
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
    if "poloxamer 188" in signal_text:
        score += 20
    if "plga:itz" in signal_text:
        score += 15
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


def collect_summary_table_candidates(table_dir: Path) -> list[dict[str, Any]]:
    manifest = load_table_manifest(table_dir)
    table_paths = sorted(table_dir.glob("*.csv"))
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
        score, row_pattern = score_summary_table(path, rows, meta)
        entries.append(
            {
                "path": path,
                "rows": rows,
                "meta": meta,
                "score": score,
                "row_pattern": row_pattern,
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
                "page_number": normalize_text(item["meta"].get("page_number")),
                "n_rows": item["meta"].get("n_rows", len(rows)),
                "n_cols": item["meta"].get("n_cols", max((len(r) for r in rows), default=0)),
                "caption_or_title": normalize_text(item["meta"].get("caption_or_title")),
                "header_keywords_hit": item["meta"].get("header_keywords_hit") or [],
                "first_data_row_preview": " | ".join(cell for cell in first_data_row if cell),
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
    if table_dir is None or not table_dir.exists():
        return {}
    manifest_path = table_dir / "tables_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict):
        tables = payload.get("tables") or []
    elif isinstance(payload, list):
        tables = payload
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
    signals = " ".join(headers + [normalize_text(item) for item in (meta.get("header_keywords_hit") or [])]).lower()
    if any(token in signals for token in ["factorial", "coded", "design", "run", "doe", "cpf", "cpva", "cplga"]):
        return "design matrix"
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
    parts = [normalize_text(part) for part in re.split(r"\n\s*\n+", text or "") if normalize_text(part)]
    return parts


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


def select_ordered_packing_paragraphs(raw_text: str, *, limit_per_kind: int = 1) -> dict[str, list[str]]:
    # This reuses the historical block-ranking idea from the earlier v7pilot packer:
    # synthesis/preparation and materials/procurement are surfaced before residual narrative.
    synthesis_blocks: list[str] = []
    materials_blocks: list[str] = []
    fallback_blocks: list[str] = []
    seen: set[str] = set()
    for paragraph in split_paragraph_blocks(raw_text):
        normalized = normalize_text(paragraph).lower()
        if not normalized or normalized in seen:
            continue
        block_type, _, score = classify_ordered_paragraph_block(paragraph)
        if block_type == "synthesis_method" and len(synthesis_blocks) < limit_per_kind and score > 0:
            synthesis_blocks.append(paragraph)
            seen.add(normalized)
            continue
        if block_type == "materials_procurement" and len(materials_blocks) < limit_per_kind and score > 0:
            materials_blocks.append(paragraph)
            seen.add(normalized)
            continue
        if len(fallback_blocks) < limit_per_kind and score > 0:
            fallback_blocks.append(paragraph)
            seen.add(normalized)
    return {
        "synthesis_blocks": synthesis_blocks,
        "materials_blocks": materials_blocks,
        "fallback_blocks": fallback_blocks,
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


def build_prompt_preview_row(
    *,
    document: dict[str, str],
    prompt_text: str,
    table_mode_value: str,
    summary_enhanced: bool,
    input_packing_mode_value: str,
    ordered_block_order: str,
) -> dict[str, Any]:
    paper_text_index = prompt_text.find("Paper text:\n")
    evidence_pack_index = prompt_text.find("Evidence pack:\n")
    table_excerpts_index = prompt_text.find("Table excerpts:\n")
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
        "prompt_length": len(prompt_text),
        "prompt_head_preview": prompt_head_preview,
        "prompt_tail_preview": prompt_tail_preview,
    }


def build_live_prompt(record: dict[str, str], text_path: Path, table_dir: Path | None, max_chars: int) -> str:
    current_table_mode = table_mode()
    summary_enhanced = summary_first_column_enhancement_enabled()
    current_input_packing_mode = input_packing_mode()
    raw_text = text_path.read_text(encoding="utf-8", errors="replace")
    table_text = render_table_text(table_dir) if current_table_mode == "full" else render_table_summary_text(table_dir)
    if ordered_input_packing_enabled():
        paper_text, ordered_block_order = build_controlled_evidence_pack(
            record=record,
            raw_text=raw_text,
            table_text=table_text,
            max_chars=max_chars,
        )
    else:
        paper_text = raw_text[:max_chars]
        ordered_block_order = "raw_prefix"
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
    if ordered_input_packing_enabled():
        controlled_input_note = (
            "- Controlled evidence packing is enabled.\n"
            "- Prompt order prioritizes synthesis/preparation blocks, then materials/procurement blocks, then table evidence, then narrative fallback.\n"
            "- Treat the evidence pack as the governed live input ordering for this run.\n"
        )
    if ordered_input_packing_enabled():
        input_block = "Evidence pack:\n" + f"{paper_text}\n"
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
    prompt_preview_path = out_dir.parent / "analysis" / PROMPT_PREVIEW_NAME
    table_selection_debug_path = out_dir.parent / "analysis" / TABLE_SELECTION_DEBUG_NAME
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
                        text_path = Path(record["text_path"])
                        if not text_path.is_absolute():
                            text_path = (PROJECT_ROOT / text_path).resolve()
                        if not text_path.exists():
                            raise FileNotFoundError(f"Missing paper text for {key}: {text_path}")
                        table_dir = resolve_tables_dir(text_path, key)
                        prompt = build_live_prompt(record, text_path, table_dir, args.max_text_chars)
                        prompt_preview_rows.append(
                            build_prompt_preview_row(
                                document={
                                    "document_key": key,
                                    "doi": record["doi"],
                                    "source_mode": "live_llm_stage2_v2",
                                },
                                prompt_text=prompt,
                                table_mode_value=current_table_mode,
                                summary_enhanced=summary_enhanced,
                                input_packing_mode_value=current_input_packing_mode,
                                ordered_block_order=ordered_block_order,
                            )
                        )
                        if current_table_mode == "summary":
                            debug_payload = build_table_selection_debug_payload(
                                document_key=key,
                                table_dir=table_dir,
                                max_tables=4,
                                summary_enhanced=summary_enhanced,
                            )
                            if debug_payload is not None:
                                table_selection_debug_rows.append(debug_payload)
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
    print(f"wrote_jsonl={jsonl_path}")
    print(f"wrote_summary={summary_path}")
    if prompt_preview_rows:
        print(f"wrote_prompt_preview={prompt_preview_path}")
    if table_selection_debug_rows:
        print(f"wrote_table_selection_debug={table_selection_debug_path}")
    print(f"wrote_raw_responses_dir={raw_dir}")


if __name__ == "__main__":
    main()
