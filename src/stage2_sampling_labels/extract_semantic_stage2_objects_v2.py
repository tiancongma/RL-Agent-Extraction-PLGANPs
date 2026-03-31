#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
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
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.model_policy import PRIMARY_DEFAULT, validate_models_or_raise
    from src.utils.paths import PROJECT_ROOT


OUTPUT_JSONL_NAME = "semantic_stage2_v2_objects.jsonl"
OUTPUT_SUMMARY_NAME = "semantic_stage2_v2_summary.tsv"
NVIDIA_HOSTED_CHAT_COMPLETIONS_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LEGACY_FIELD_ALIASES = {"plga_mw_kDa": "polymer_mw_kDa"}
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


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_token(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def canonical_field_name(field_name: str) -> str:
    return LEGACY_FIELD_ALIASES.get(field_name, field_name)


def safe_json_load(text: str) -> dict[str, Any]:
    cleaned = normalize_text(text.replace("\r\n", "\n").replace("\r", "\n"))
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.replace("\t", " ")
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
    raise ValueError("Could not parse JSON response.")


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
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing in environment.")
    genai.configure(api_key=api_key)
    if not str(model or "").strip():
        raise RuntimeError("Gemini model name is empty.")


def call_gemini(model: str, prompt: str, retries: int, sleep_sec: float) -> str:
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
            time.sleep(sleep_sec)
    raise last_err or RuntimeError("Gemini call failed.")


def call_nvidia_hosted(model: str, prompt: str, retries: int, sleep_sec: float) -> str:
    load_dotenv()
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
            time.sleep(sleep_sec * (attempt + 1))
    raise last_err or RuntimeError("NVIDIA hosted API call failed.")


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


def build_live_prompt(record: dict[str, str], text_path: Path, table_dir: Path | None, max_chars: int) -> str:
    paper_text = text_path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    table_text = render_table_text(table_dir)
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
    }
    return (
        "You are extracting Stage2 v2 semantic objects for a governed comparator slice.\n"
        "Rules:\n"
        "- Preserve ambiguity explicitly.\n"
        "- Emit object-first outputs only.\n"
        "- Do not perform relation resolution, inheritance closure, or final-row materialization.\n"
        "- Do not force DOE rows if the paper only reports factors but not clear formulation boundaries.\n"
        "- Return valid JSON only.\n\n"
        f"Paper key: {record['key']}\n"
        f"DOI: {record['doi']}\n"
        f"Title: {record['title']}\n\n"
        "Return JSON with exactly these top-level keys:\n"
        f"{json.dumps(schema, ensure_ascii=True, indent=2)}\n\n"
        "Paper text:\n"
        f"{paper_text}\n\n"
        "Table excerpts:\n"
        f"{table_text}\n"
    )


def find_legacy_raw_response(raw_dir: Path, key: str) -> Path:
    matches = sorted(raw_dir.glob(f"*_{key}_*.txt"))
    if not matches:
        raise FileNotFoundError(f"Legacy raw response not found for {key} under {raw_dir}")
    return matches[0]


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
    parsed = safe_json_load(raw_response_text)
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
    return {
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


def normalize_live_document(record: dict[str, str], parsed: dict[str, Any], raw_response_path: Path) -> dict[str, Any]:
    text_path = Path(record["text_path"])
    if not text_path.is_absolute():
        text_path = (PROJECT_ROOT / text_path).resolve()
    table_dir = resolve_tables_dir(text_path, record["key"])
    source_table_files = []
    if table_dir and table_dir.exists():
        source_table_files = [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in sorted(table_dir.glob("*.csv"))]
    return {
        "document_key": record["key"],
        "doi": record["doi"],
        "title": record["title"],
        "source_mode": "live_llm_stage2_v2",
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
    }


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
    return {
        "document_key": document["document_key"],
        "doi": document["doi"],
        "source_mode": document["source_mode"],
        "formulation_count": len(document.get("formulation_candidates", [])),
        "component_count": len(document.get("component_candidates", [])),
        "variable_count": len(document.get("variable_candidates", [])),
        "measurement_count": len(document.get("measurement_candidates", [])),
        "relation_hint_count": len(document.get("relation_hints", [])),
        "evidence_span_count": len(document.get("evidence_spans", [])),
        "unassigned_observation_count": len(document.get("unassigned_observations", [])),
        "ph_variable_count": sum(1 for name in variable_names if name in PH_TOKENS),
        "doe_factor_count": sum(1 for name in variable_names if name in DOE_TOKENS),
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
        help="Use saved historical raw responses or call a live model.",
    )
    parser.add_argument(
        "--legacy-raw-responses-dir",
        default="",
        help="Directory containing saved historical raw responses for replay mode.",
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
    summary_rows: list[dict[str, Any]] = []

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            key = normalize_text(record.get("key"))
            if not key:
                continue
            if "text_path" not in record or not normalize_text(record.get("text_path")):
                raise ValueError(f"Manifest row for {key} is missing text_path.")

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
                if args.llm_backend == "gemini":
                    raw_text = call_gemini(args.model, prompt, args.request_retries, args.retry_sleep_sec)
                else:
                    raw_text = call_nvidia_hosted(args.model, prompt, args.request_retries, args.retry_sleep_sec)
                raw_copy_path = raw_dir / f"{key}__stage2_v2_raw_response.json"
                raw_copy_path.write_text(raw_text, encoding="utf-8")
                parsed = safe_json_load(raw_text)
                document = normalize_live_document(record, parsed, raw_copy_path)

            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
            summary_rows.append(summary_row(document))

    write_tsv(
        summary_path,
        summary_rows,
        [
            "document_key",
            "doi",
            "source_mode",
            "formulation_count",
            "component_count",
            "variable_count",
            "measurement_count",
            "relation_hint_count",
            "evidence_span_count",
            "unassigned_observation_count",
            "ph_variable_count",
            "doe_factor_count",
            "pdi_measurement_present",
            "zeta_measurement_present",
            "multi_component_formulation_count",
        ],
    )
    print(f"wrote_jsonl={jsonl_path}")
    print(f"wrote_summary={summary_path}")
    print(f"wrote_raw_responses_dir={raw_dir}")


if __name__ == "__main__":
    main()
