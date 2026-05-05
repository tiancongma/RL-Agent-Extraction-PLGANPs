#!/usr/bin/env python3
from __future__ import annotations

"""
S5-3a prompt-construction freeze boundary for Stage5 LLM direct-value candidates.

This supporting runner consumes explicit fixed Stage5/source inputs, materializes
inspectable prompt artifacts, and stops before any live LLM/API call.
"""

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.stage5_benchmark.compare_layer3_values_to_gt_v1 import SYSTEM_FIELD_MAP
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution from repo root.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage5_benchmark.compare_layer3_values_to_gt_v1 import SYSTEM_FIELD_MAP

ENTRYPOINT = "src/stage5_benchmark/build_s5_3a_llm_value_prompts_v1.py"
STAGE_BOUNDARY = "S5-3a"
BOUNDARY_CLASS = "diagnostic/supporting_boundary"
BENCHMARK_VALID_STATUS = "no"

PROMPTS_JSONL_NAME = "s5_3a_llm_direct_value_prompts_v1.jsonl"
PROMPT_AUDIT_TSV_NAME = "s5_3a_llm_direct_value_prompt_audit_v1.tsv"
INPUT_MANIFEST_TSV_NAME = "s5_3a_llm_direct_value_input_manifest_v1.tsv"
SOURCE_EVIDENCE_AUDIT_TSV_NAME = "s5_3a_source_evidence_audit_v1.tsv"
PROMPT_TEMPLATE_NAME = "s5_3a_llm_direct_value_prompt_template_v1.txt"
RUN_CONTEXT_NAME = "RUN_CONTEXT.md"

PAPER_KEY_ALIASES = ("paper_key", "key", "article_key", "source_key")
FORMULATION_ID_ALIASES = (
    "final_formulation_id",
    "formulation_id",
    "article_formulation_id",
    "parent_core_row_id",
)

DEFAULT_TARGET_MODE = "missing_system"

# S5-3 runs after S5-2 deterministic materialization. Its default target
# surface is therefore gap-filling only: target S5-3-responsible fields that
# remain unfilled in the current fixed Stage5 rows, not the complete Layer3
# field map and not row-local mechanical table-cell fields that S5-2 already
# lawfully owns.
S5_2_MECHANICAL_TABLE_CELL_FIELDS = {
    "drug_name",
    "polymer_name",
    "polymer_mass_mg",
    "drug_mass_mg",
    "O_volume_mL",
    "external_aqueous_phase_volume_mL",
    "surfactant_name",
    "particle_size_nm",
    "pdi",
    "zeta_mV",
    "ee_percent",
}
S5_3_RESPONSIBLE_FIELDS = [
    field_name
    for field_name in SYSTEM_FIELD_MAP.keys()
    if field_name not in S5_2_MECHANICAL_TABLE_CELL_FIELDS
]
TARGET_FIELDS = list(S5_3_RESPONSIBLE_FIELDS)

FIELD_VALUE_COLUMN_ALIASES = {
    "method_type": ["method_type_value_text", "preparation_method", "emul_method_value_text", "emul_type_value_text"],
    "drug_name": ["drug_name_value_text"],
    "polymer_name": ["polymer_name_value_text", "polymer_name_raw", "polymer_identity_final", "polymer_identity"],
    "polymer_mass_mg": ["polymer_mass_mg_value_text"],
    "polymer_mw_raw": ["polymer_mw_raw_value_text", "polymer_mw_kDa_value_text"],
    "polymer_mw_kDa": ["polymer_mw_kDa_value_text"],
    "la_ga_ratio_raw": ["la_ga_ratio_raw_value_text", "la_ga_ratio_value_text"],
    "la_ga_ratio_normalized": ["la_ga_ratio_normalized_value_text", "la_ga_ratio_value_text"],
    "solvent_name": ["solvent_name_value_text", "organic_solvent_value_text"],
    "O_volume_mL": ["O_volume_mL_value_text", "organic_phase_volume_mL_value_text"],
    "external_aqueous_phase_volume_mL": ["external_aqueous_phase_volume_mL_value_text"],
    "surfactant_name": ["surfactant_name_value_text"],
    "emulsifier_stabilizer_name": ["emulsifier_stabilizer_name_value_text", "surfactant_name_value_text", "stabilizer_name_value_text"],
    "emulsifier_stabilizer_concentration_value": ["emulsifier_stabilizer_concentration_value_value_text", "surfactant_concentration_value_value_text", "surfactant_concentration_text_value_text"],
    "emulsifier_stabilizer_concentration_unit": ["emulsifier_stabilizer_concentration_unit_value_text", "surfactant_concentration_unit_value_text"],
    "particle_size_nm": ["particle_size_nm_value_text", "size_nm_value_text"],
    "pdi": ["pdi_value_text"],
    "zeta_mV": ["zeta_mV_value_text"],
    "ee_percent": ["ee_percent_value_text", "encapsulation_efficiency_percent_value_text"],
    "dl_percent": ["dl_percent_value_text", "drug_loading_percent_value_text"],
    "lc_percent": ["lc_percent_value_text", "loading_content_percent_value_text", "loading_capacity_percent_value_text"],
    "pH_raw": ["pH_raw_value_text"],
    "phase_ratio_raw": ["phase_ratio_raw_value_text"],
}
for _field_name, _field_spec in SYSTEM_FIELD_MAP.items():
    if _field_name not in FIELD_VALUE_COLUMN_ALIASES:
        _column = str(_field_spec.get("column") or "").strip()
        FIELD_VALUE_COLUMN_ALIASES[_field_name] = [
            name
            for name in [_column, _field_name, f"{_field_name}_value_text"]
            if name
        ]

EMPTY_SYSTEM_VALUES = {"", "unknown", "[]", "{}", "null", "none", "n/a", "na"}
RESOLVED_ABSENCE_MISSING_REASONS = {
    "not_reported",
    "not_applicable",
    "not_applicable_to_formulation",
    "not_in_source",
}

PROMPT_TEMPLATE = """You are gap-filling omitted S5-3-responsible direct source-backed formulation values for a fixed Stage5 row universe after S5-2 deterministic materialization.

Return ONLY valid JSON with this exact top-level shape:
{
  "paper_key": "<paper key>",
  "candidates": [
    {
      "paper_key": "<paper key>",
      "formulation_id": "<one exact final_formulation_id from FIXED_STAGE5_ROWS>",
      "field_name": "<one target field>",
      "value_text": "<value exactly as supported by source, without unit if a separate unit_text is appropriate>",
      "unit_text": "<unit text, or empty string>",
      "raw_cell_text": "<raw table cell/span text>",
      "direct_or_derived": "direct|derived|ambiguous",
      "evidence_type": "table_cell|table_row|method_paragraph|global_preparation_text|other",
      "evidence_scope": "row_local|table_row|table_scoped_unique|paper_global_unique|ambiguous|unknown",
      "source_file": "<source path shown in prompt>",
      "source_table_id": "<table id if applicable, else empty>",
      "source_row_id": "<row id/number if applicable, else empty>",
      "source_quote": "<short exact quote copied from SOURCE_EVIDENCE>",
      "confidence": "high|medium|low",
      "needs_review": "yes|no",
      "llm_rationale_short": "<brief reason>"
    }
  ]
}

Rules:
- Do not create, remove, rename, or merge formulation rows.
- Use only formulation_id values listed in FIXED_STAGE5_ROWS.
- TARGET_FIELDS is the S5-3-responsible gap surface after S5-2 deterministic materialization: fields already filled in the current Stage5 rows should not be re-extracted.
- TARGET_FIELDS_BY_FORMULATION_ID is the authoritative row-specific gap list; emit candidates only for fields listed under that exact formulation_id.
- Extract explicitly source-backed direct values only for the listed row-specific target fields.
- Do not re-extract S5-2-owned mechanical row-local table-cell fields, and do not re-extract values already present in the current Stage5 row context.
- Emit direct values only when the source explicitly reports the value or an article-native table/header/row label explicitly binds it.
- Do not calculate values. If a value would require arithmetic, set direct_or_derived="derived" and needs_review="yes"; do not present it as direct.
- Do not use the current Stage5 system values as source authority. They are row context only.
- Preserve exact source_quote text and exact source_file/table/row provenance.
- If row scope is unclear, set evidence_scope="ambiguous" and needs_review="yes".
- Keep output compact; omit candidates you cannot source with an exact quote.

TARGET_FIELDS_UNION:
{target_fields_json}

TARGET_FIELDS_BY_FORMULATION_ID:
{target_fields_by_formulation_id_json}

PAPER_KEY:
{paper_key}

FIXED_STAGE5_ROWS:
{fixed_rows_json}

SOURCE_EVIDENCE:
{source_evidence}
"""


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _first_present(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = _clean(row.get(name))
        if value:
            return value
    return ""


def resolve_existing_path(value: Path, role: str, *, directory_ok: bool = False) -> Path:
    path = value.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required explicit {role} does not exist: {path}")
    if not directory_ok and not path.is_file():
        raise FileNotFoundError(f"Required explicit {role} is not a file: {path}")
    return path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def write_tsv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _clean(row.get(column)) for column in columns})


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_text_path(paper_key: str) -> Path | None:
    candidates = [
        Path("data/cleaned/content/text") / f"{paper_key}.pdf.txt",
        Path("data/cleaned/content/text") / f"{paper_key}.html.txt",
        Path("data/cleaned/content/text") / f"{paper_key}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def infer_table_dir(paper_key: str) -> Path | None:
    candidate = Path("data/cleaned/goren_2025/tables") / paper_key
    return candidate.resolve() if candidate.exists() else None


def infer_stage2_evidence_blocks_path(paper_key: str, root: Path | None) -> Path | None:
    if root is None:
        return None
    candidates = [
        root / "evidence_blocks" / paper_key / "evidence_blocks_v1.json",
        root / paper_key / "evidence_blocks_v1.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def infer_stage2_normalized_table_payloads_path(paper_key: str, root: Path | None) -> Path | None:
    if root is None:
        return None
    candidates = [
        root / "normalized_table_payloads" / paper_key / "normalized_table_payloads_v1.json",
        root / paper_key / "normalized_table_payloads_v1.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def read_source_inventory(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    rows = read_tsv(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        paper_key = _first_present(row, PAPER_KEY_ALIASES)
        if not paper_key:
            continue
        out[paper_key] = dict(row)
    return out


def select_final_rows(final_rows: list[dict[str, str]], paper_key: str, max_rows: int, row_offset: int = 0) -> list[dict[str, str]]:
    selected = [row for row in final_rows if _first_present(row, PAPER_KEY_ALIASES) == paper_key]
    if not selected:
        raise ValueError(f"No final-table rows found for paper_key={paper_key}")
    if row_offset < 0:
        raise ValueError("row_offset must be >= 0")
    if row_offset:
        selected = selected[row_offset:]
    if max_rows > 0:
        selected = selected[:max_rows]
    if not selected:
        raise ValueError(f"No final-table rows selected for paper_key={paper_key} after row_offset={row_offset} max_rows={max_rows}")
    return selected


def compact_final_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    preferred = [
        "final_formulation_id",
        "formulation_id",
        "representative_source_formulation_id",
        "raw_formulation_label",
        "representative_source_raw_formulation_label",
        "source_candidate_labels",
    ]
    out: list[dict[str, str]] = []
    for row in rows:
        item = {field: _clean(row.get(field)) for field in preferred if _clean(row.get(field))}
        item["paper_key"] = _first_present(row, PAPER_KEY_ALIASES)
        item["formulation_id_for_candidate_output"] = _first_present(row, FORMULATION_ID_ALIASES)
        out.append(item)
    return out


def _system_value_present(row: dict[str, str], field_name: str) -> bool:
    for column in FIELD_VALUE_COLUMN_ALIASES.get(field_name, [field_name, f"{field_name}_value_text"]):
        value = _clean(row.get(column))
        if value and value.lower() not in EMPTY_SYSTEM_VALUES:
            return True
    return False


def _system_field_resolved_absent(row: dict[str, str], field_name: str) -> bool:
    missing_reason = _clean(row.get(f"{field_name}_missing_reason")).lower()
    return missing_reason in RESOLVED_ABSENCE_MISSING_REASONS


def _system_field_has_final_table_surface(row: dict[str, str], field_name: str) -> bool:
    aliases = FIELD_VALUE_COLUMN_ALIASES.get(field_name, [field_name, f"{field_name}_value_text"])
    if any(column in row for column in aliases):
        return True
    if f"{field_name}_missing_reason" in row:
        return True
    return False


def _system_field_needs_s5_3_gap_fill(row: dict[str, str], field_name: str) -> bool:
    if not _system_field_has_final_table_surface(row, field_name):
        return False
    if _system_value_present(row, field_name):
        return False
    if _system_field_resolved_absent(row, field_name):
        return False
    return True


def _candidate_target_fields(requested_fields: list[str]) -> list[str]:
    requested = [field for field in requested_fields if _clean(field)]
    unknown = [field for field in requested if field not in TARGET_FIELDS]
    if unknown:
        raise ValueError(f"Unknown --target-field value(s): {unknown}")
    return requested or list(TARGET_FIELDS)


def select_target_fields_by_formulation_id(rows: list[dict[str, str]], requested_fields: list[str], target_mode: str) -> dict[str, list[str]]:
    candidate_fields = _candidate_target_fields(requested_fields)
    target_by_row: dict[str, list[str]] = {}
    for row in rows:
        formulation_id = _first_present(row, FORMULATION_ID_ALIASES)
        if target_mode == "all":
            fields = list(candidate_fields)
        else:
            fields = [field for field in candidate_fields if _system_field_needs_s5_3_gap_fill(row, field)]
        target_by_row[formulation_id] = fields
    return target_by_row


def select_target_fields_for_rows(rows: list[dict[str, str]], requested_fields: list[str], target_mode: str) -> list[str]:
    candidate_fields = _candidate_target_fields(requested_fields)
    target_by_row = select_target_fields_by_formulation_id(rows, requested_fields, target_mode)
    return [field for field in candidate_fields if any(field in fields for fields in target_by_row.values())]


def extract_source_table_ids(rows: list[dict[str, str]]) -> set[str]:
    table_ids: set[str] = set()
    for row in rows:
        for field in ["formulation_id", "representative_source_formulation_id", "final_formulation_id", "source_candidate_ids"]:
            for match in re.finditer(r"__table_(\d+)__", _clean(row.get(field))):
                table_ids.add(f"table_{int(match.group(1)):02d}")
    return table_ids


def extract_relevant_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    anchors = [
        "Preparation",
        "Materials and methods",
        "Experimental",
        "Nanoparticle preparation",
        "Preparation of Polymeric Nanoparticles",
        "Characterization",
        "Characterization of the Nanocarriers",
        "Encapsulation efficiency",
        "Drug loading",
        "Table 1",
        "Table 2",
        "Table 3",
    ]
    chunks: list[str] = []
    for anchor in anchors:
        idx = text.lower().find(anchor.lower())
        if idx >= 0:
            start = max(0, idx - 1200)
            end = min(len(text), idx + 6500)
            chunks.append(text[start:end])
    if not chunks:
        chunks.append(text[:max_chars])
    merged = "\n\n--- EXCERPT BREAK ---\n\n".join(chunks)
    return merged[:max_chars]


def read_table_evidence(table_dir: Path | None, max_chars: int, allowed_table_ids: set[str] | None = None) -> str:
    if table_dir is None or not table_dir.exists():
        return ""
    parts: list[str] = []
    for path in sorted(table_dir.glob("*.csv")):
        text = path.read_text(encoding="utf-8", errors="replace")
        include = False
        if allowed_table_ids is None:
            include = True
        elif allowed_table_ids:
            include = any(f"__{table_id}__" in path.name for table_id in allowed_table_ids)
        lowered = text.lower()
        if allowed_table_ids == set() and "formulation" in lowered and ("surfactant" in lowered or "characterization" in lowered or "ee" in lowered):
            include = True
        if include:
            parts.append(f"SOURCE_TABLE_FILE: {path}\n{text}")
        if sum(len(p) for p in parts) >= max_chars:
            break
    return "\n\n".join(parts)[:max_chars]


def _is_s5_3_prompt_noise_line(line: str) -> bool:
    lowered = line.strip().lower()
    if not lowered:
        return False
    noise_patterns = (
        "downloaded from http",
        "downloaded from https",
        "academic.oup.com",
        "wiley online library",
        "journals & books",
        "download full issue",
        "view pdf",
        "submit your manuscript",
        "unauthenticated",
    )
    return any(pattern in lowered for pattern in noise_patterns)


def clean_stage2_evidence_text_for_s5_3_prompt(text: str) -> tuple[str, int]:
    cleaned_lines: list[str] = []
    removed = 0
    for raw_line in text.splitlines():
        if _is_s5_3_prompt_noise_line(raw_line):
            removed += 1
            continue
        cleaned_lines.append(raw_line)
    return _clean("\n".join(cleaned_lines)), removed


def _stage2_block_is_suppressible(block: dict[str, Any]) -> bool:
    block_type = _clean(block.get("block_type")).lower()
    evidence_kind = _clean(block.get("evidence_kind")).lower()
    source_type = _clean(block.get("source_type")).lower()
    flags = [str(flag).lower() for flag in (block.get("noise_flags") or []) + (block.get("quality_flags") or [])]
    guardrails = [str(flag).lower() for flag in (block.get("primary_guardrail_reason") or [])]
    text = _clean(block.get("text_content")).lower()
    reason = " ".join([block_type, evidence_kind, source_type, " ".join(flags), " ".join(guardrails)])
    suppress_markers = (
        "reference_like_content",
        "front_matter",
        "suppressible_noncanonical_context",
        "residual_noise",
    )
    if any(marker in reason for marker in suppress_markers):
        return True
    if "table_type_non_formulation_table" in guardrails and "very_low_numeric_density_without_header_keywords" in guardrails:
        return True
    noisy_table_markers = (
        "department of",
        "correspondence",
        "download full issue",
        "journals & books",
        "view pdf",
        "submit your manuscript",
        "wiley online library",
    )
    return block_type == "table" and any(marker in text for marker in noisy_table_markers)


def build_stage2_clean_evidence(
    paper_key: str,
    evidence_blocks_path: Path | None,
    normalized_payloads_path: Path | None,
    max_text_chars: int,
    max_table_chars: int,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if evidence_blocks_path is None or not evidence_blocks_path.exists():
        return "", [], {"evidence_source_mode": "raw_fallback", "stage2_evidence_blocks_path": "", "stage2_normalized_table_payloads_path": str(normalized_payloads_path or "")}
    payload = json.loads(evidence_blocks_path.read_text(encoding="utf-8"))
    blocks = payload.get("evidence_blocks") or []
    text_parts: list[str] = []
    table_parts: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    skipped = 0
    removed_noise_lines_total = 0
    for block in blocks:
        raw_block_text = _clean(block.get("text_content"))
        block_text, removed_noise_lines = clean_stage2_evidence_text_for_s5_3_prompt(raw_block_text)
        removed_noise_lines_total += removed_noise_lines
        if not block_text:
            continue
        suppressible = _stage2_block_is_suppressible(block)
        if suppressible:
            skipped += 1
            continue
        block_id = _clean(block.get("block_id"))
        block_type = _clean(block.get("block_type"))
        evidence_kind = _clean(block.get("evidence_kind"))
        source_type = _clean(block.get("source_type"))
        origin = _clean(block.get("origin_locator"))
        table_id = _clean(block.get("table_id"))
        header = "\n".join([
            f"BEGIN_STAGE2_EVIDENCE_BLOCK: {block_id}",
            f"SOURCE_FILE: {origin}",
            f"BLOCK_TYPE: {block_type}",
            f"EVIDENCE_KIND: {evidence_kind}",
            f"SOURCE_TYPE: {source_type}",
            f"SOURCE_TABLE_ID: {table_id}",
        ])
        rendered = f"{header}\n{block_text}\nEND_STAGE2_EVIDENCE_BLOCK: {block_id}"
        if block.get("is_table_derived") or evidence_kind == "table" or source_type.startswith("table"):
            table_parts.append(rendered)
        else:
            text_parts.append(rendered)
        audit_rows.append({
            "paper_key": paper_key,
            "evidence_source_mode": "stage2_clean_evidence_blocks",
            "block_id": block_id,
            "block_type": block_type,
            "evidence_kind": evidence_kind,
            "source_type": source_type,
            "origin_locator": origin,
            "table_id": table_id,
            "char_count": len(block_text),
            "removed_prompt_noise_lines": removed_noise_lines,
            "included": "yes",
            "skip_reason": "",
        })
    text_joined = "\n\n".join(text_parts)[:max_text_chars]
    table_joined = "\n\n".join(table_parts)[:max_table_chars]
    sections = []
    if text_joined:
        sections.append("SOURCE_TEXT_AUTHORITY: stage2_clean_evidence_blocks\n" + text_joined)
    if table_joined:
        table_header = "SOURCE_TABLE_AUTHORITY: stage2_evidence_blocks_summary"
        if normalized_payloads_path and normalized_payloads_path.exists():
            table_header += f"\nNORMALIZED_TABLE_PAYLOADS: {normalized_payloads_path}"
        sections.append(table_header + "\n" + table_joined)
    metadata = {
        "evidence_source_mode": "stage2_clean_evidence_blocks",
        "stage2_evidence_blocks_path": str(evidence_blocks_path),
        "stage2_normalized_table_payloads_path": str(normalized_payloads_path or ""),
        "stage2_contract_version": _clean(payload.get("contract_version")),
        "stage2_segmentation_profile": _clean(payload.get("segmentation_profile")),
        "stage2_selection_mode": _clean(payload.get("selection_mode")),
        "stage2_blocks_total": len(blocks),
        "stage2_blocks_included": len(audit_rows),
        "stage2_blocks_skipped_suppressible": skipped,
        "stage2_prompt_noise_lines_removed": removed_noise_lines_total,
    }
    return "\n\n=====\n\n".join(sections), audit_rows, metadata


def build_source_evidence(
    paper_key: str,
    text_path: Path | None,
    table_dir: Path | None,
    max_text_chars: int,
    max_table_chars: int,
    allowed_table_ids: set[str] | None = None,
    evidence_blocks_path: Path | None = None,
    normalized_payloads_path: Path | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    clean_evidence, audit_rows, metadata = build_stage2_clean_evidence(
        paper_key,
        evidence_blocks_path,
        normalized_payloads_path,
        max_text_chars,
        max_table_chars,
    )
    if clean_evidence:
        return clean_evidence, audit_rows, metadata

    sections: list[str] = []
    fallback_rows: list[dict[str, Any]] = []
    if text_path and text_path.exists():
        text = text_path.read_text(encoding="utf-8", errors="replace")
        excerpt = extract_relevant_text(text, max_text_chars)
        sections.append(f"SOURCE_TEXT_FALLBACK_RAW_CLEANED_FILE: {text_path}\nBEGIN_RAW_FALLBACK_TEXT\n" + excerpt + "\nEND_RAW_FALLBACK_TEXT")
        fallback_rows.append({
            "paper_key": paper_key,
            "evidence_source_mode": "raw_text_fallback",
            "block_id": "raw_text_fallback",
            "block_type": "raw_text_fallback",
            "evidence_kind": "text",
            "source_type": "raw_cleaned_text_file",
            "origin_locator": str(text_path),
            "table_id": "",
            "char_count": len(excerpt),
            "included": "yes",
            "skip_reason": "stage2_evidence_blocks_missing",
        })
    table_text = read_table_evidence(table_dir, max_table_chars, allowed_table_ids)
    if table_text:
        sections.append("SOURCE_TABLE_FALLBACK_RAW_CSV: compatibility_debt\n" + table_text)
        fallback_rows.append({
            "paper_key": paper_key,
            "evidence_source_mode": "raw_table_csv_fallback",
            "block_id": "raw_table_csv_fallback",
            "block_type": "raw_table_csv_fallback",
            "evidence_kind": "table",
            "source_type": "raw_stage1_table_csv",
            "origin_locator": str(table_dir or ""),
            "table_id": "__ALL_SOURCE_TABLES__" if allowed_table_ids is None else json.dumps(sorted(allowed_table_ids), ensure_ascii=False),
            "char_count": len(table_text),
            "included": "yes",
            "skip_reason": "stage2_table_authority_missing",
        })
    if not sections:
        raise ValueError(f"No source evidence found for {paper_key}; pass Stage2 evidence roots or --source-text-path/--source-table-dir")
    metadata = {
        "evidence_source_mode": "raw_fallback",
        "stage2_evidence_blocks_path": str(evidence_blocks_path or ""),
        "stage2_normalized_table_payloads_path": str(normalized_payloads_path or ""),
        "fallback_raw_text_path": str(text_path or ""),
        "fallback_raw_table_dir": str(table_dir or ""),
    }
    return "\n\n=====\n\n".join(sections), fallback_rows, metadata


def source_path_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": "", "exists": "no", "sha256": ""}
    return {"path": str(path), "exists": "yes" if path.exists() else "no", "sha256": sha256_file(path) if path.is_file() else "directory"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build frozen S5-3a prompt artifacts and stop before live LLM calls.")
    parser.add_argument("--final-table-tsv", type=Path, required=True)
    parser.add_argument("--decision-trace-tsv", type=Path, required=True)
    parser.add_argument("--scope-manifest-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--paper-key", required=True)
    parser.add_argument("--source-inventory-tsv", type=Path, default=None)
    parser.add_argument("--source-text-path", type=Path, default=None)
    parser.add_argument("--source-table-dir", type=Path, default=None)
    parser.add_argument("--stage2-semantic-objects-dir", type=Path, default=None)
    parser.add_argument("--stage2-evidence-blocks-root", type=Path, default=None)
    parser.add_argument("--stage2-normalized-table-payloads-root", type=Path, default=None)
    parser.add_argument("--max-final-rows", type=int, default=0)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--max-text-chars", type=int, default=70000)
    parser.add_argument("--max-table-chars", type=int, default=60000)
    parser.add_argument("--target-mode", choices=["missing_system", "all"], default=DEFAULT_TARGET_MODE)
    parser.add_argument("--target-field", action="append", default=[], dest="target_fields")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    final_table_tsv = resolve_existing_path(args.final_table_tsv, "final-table-tsv")
    decision_trace_tsv = resolve_existing_path(args.decision_trace_tsv, "decision-trace-tsv")
    scope_manifest_tsv = resolve_existing_path(args.scope_manifest_tsv, "scope-manifest-tsv")
    source_inventory_tsv = resolve_existing_path(args.source_inventory_tsv, "source-inventory-tsv") if args.source_inventory_tsv else None
    paper_key = _clean(args.paper_key)
    if not paper_key:
        raise ValueError("--paper-key is required")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_inventory = read_source_inventory(source_inventory_tsv)
    semantic_objects_dir = resolve_existing_path(args.stage2_semantic_objects_dir, "stage2-semantic-objects-dir", directory_ok=True) if args.stage2_semantic_objects_dir else None
    evidence_blocks_root_arg = resolve_existing_path(args.stage2_evidence_blocks_root, "stage2-evidence-blocks-root", directory_ok=True) if args.stage2_evidence_blocks_root else None
    normalized_payloads_root_arg = resolve_existing_path(args.stage2_normalized_table_payloads_root, "stage2-normalized-table-payloads-root", directory_ok=True) if args.stage2_normalized_table_payloads_root else None
    evidence_blocks_root = evidence_blocks_root_arg or (semantic_objects_dir / "evidence_blocks" if semantic_objects_dir else None)
    normalized_payloads_root = normalized_payloads_root_arg or (semantic_objects_dir / "normalized_table_payloads" if semantic_objects_dir else None)
    evidence_blocks_path = infer_stage2_evidence_blocks_path(paper_key, evidence_blocks_root)
    normalized_payloads_path = infer_stage2_normalized_table_payloads_path(paper_key, normalized_payloads_root)
    text_path = resolve_existing_path(args.source_text_path, "source-text-path") if args.source_text_path else infer_text_path(paper_key)
    table_dir = resolve_existing_path(args.source_table_dir, "source-table-dir", directory_ok=True) if args.source_table_dir else infer_table_dir(paper_key)

    final_rows = select_final_rows(read_tsv(final_table_tsv), paper_key, args.max_final_rows, args.row_offset)
    target_fields_by_formulation_id = select_target_fields_by_formulation_id(final_rows, args.target_fields, args.target_mode)
    target_fields = select_target_fields_for_rows(final_rows, args.target_fields, args.target_mode)
    fixed_rows = compact_final_rows(final_rows)
    allowed_table_ids = None if args.target_mode == "all" and not args.target_fields else extract_source_table_ids(final_rows)
    source_table_ids_for_manifest = ["__ALL_SOURCE_TABLES__"] if allowed_table_ids is None else sorted(allowed_table_ids)
    source_evidence, evidence_audit_rows, evidence_metadata = build_source_evidence(
        paper_key,
        text_path,
        table_dir,
        args.max_text_chars,
        args.max_table_chars,
        allowed_table_ids,
        evidence_blocks_path,
        normalized_payloads_path,
    )
    prompt_text = (
        PROMPT_TEMPLATE
        .replace("{target_fields_json}", json.dumps(target_fields, ensure_ascii=False))
        .replace("{target_fields_by_formulation_id_json}", json.dumps(target_fields_by_formulation_id, ensure_ascii=False, indent=2))
        .replace("{paper_key}", paper_key)
        .replace("{fixed_rows_json}", json.dumps(fixed_rows, ensure_ascii=False, indent=2))
        .replace("{source_evidence}", source_evidence)
    )
    prompt_hash = sha256_text(prompt_text)
    input_hash_payload = {
        "final_table_tsv": source_path_metadata(final_table_tsv),
        "decision_trace_tsv": source_path_metadata(decision_trace_tsv),
        "scope_manifest_tsv": source_path_metadata(scope_manifest_tsv),
        "source_inventory_tsv": source_path_metadata(source_inventory_tsv),
        "stage2_semantic_objects_dir": source_path_metadata(semantic_objects_dir),
        "stage2_evidence_blocks_path": source_path_metadata(evidence_blocks_path),
        "stage2_normalized_table_payloads_path": source_path_metadata(normalized_payloads_path),
        "source_text_path": source_path_metadata(text_path),
        "source_table_dir": source_path_metadata(table_dir),
        "paper_key": paper_key,
        "row_offset": args.row_offset,
        "max_final_rows": args.max_final_rows,
        "target_mode": args.target_mode,
        "target_fields": target_fields,
        "target_fields_by_formulation_id": target_fields_by_formulation_id,
        "source_table_ids": source_table_ids_for_manifest,
        "evidence_source_mode": evidence_metadata.get("evidence_source_mode", ""),
        "evidence_metadata": evidence_metadata,
        "final_row_ids": [_clean(row.get("formulation_id_for_candidate_output")) for row in fixed_rows],
    }
    input_artifact_hash = sha256_text(json.dumps(input_hash_payload, sort_keys=True, ensure_ascii=False))

    prompts_jsonl = out_dir / PROMPTS_JSONL_NAME
    prompt_template = out_dir / PROMPT_TEMPLATE_NAME
    prompt_audit = out_dir / PROMPT_AUDIT_TSV_NAME
    input_manifest = out_dir / INPUT_MANIFEST_TSV_NAME
    source_evidence_audit = out_dir / SOURCE_EVIDENCE_AUDIT_TSV_NAME
    run_context = out_dir / RUN_CONTEXT_NAME

    prompt_template.write_text(PROMPT_TEMPLATE, encoding="utf-8")
    prompt_id_suffix = f"rows_{args.row_offset + 1:03d}_{args.row_offset + len(final_rows):03d}"
    prompt_row = {
        "stage_boundary": STAGE_BOUNDARY,
        "paper_key": paper_key,
        "prompt_id": f"{paper_key}__{prompt_id_suffix}__s5_3a_direct_value_prompt_v1",
        "prompt_text": prompt_text,
        "prompt_sha256": prompt_hash,
        "input_artifact_hash": input_artifact_hash,
        "final_row_count": len(final_rows),
        "row_offset": args.row_offset,
        "max_final_rows": args.max_final_rows,
        "target_mode": args.target_mode,
        "target_fields": json.dumps(target_fields, ensure_ascii=False),
        "target_fields_by_formulation_id": json.dumps(target_fields_by_formulation_id, ensure_ascii=False),
        "source_table_ids": json.dumps(source_table_ids_for_manifest, ensure_ascii=False),
        "evidence_source_mode": evidence_metadata.get("evidence_source_mode", ""),
        "stage2_evidence_blocks_path": str(evidence_blocks_path or ""),
        "stage2_normalized_table_payloads_path": str(normalized_payloads_path or ""),
        "source_text_path": str(text_path or ""),
        "source_table_dir": str(table_dir or ""),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    prompts_jsonl.write_text(json.dumps(prompt_row, ensure_ascii=False) + "\n", encoding="utf-8")

    write_tsv(
        prompt_audit,
        [
            "stage_boundary",
            "paper_key",
            "prompt_id",
            "prompt_sha256",
            "input_artifact_hash",
            "prompt_char_count",
            "final_row_count",
            "row_offset",
            "max_final_rows",
            "target_mode",
            "target_fields",
            "target_fields_by_formulation_id",
            "source_table_ids",
            "evidence_source_mode",
            "stage2_evidence_blocks_path",
            "stage2_normalized_table_payloads_path",
            "source_text_path",
            "source_table_dir",
            "prompt_status",
        ],
        [
            {
                **{k: prompt_row[k] for k in ["stage_boundary", "paper_key", "prompt_id", "prompt_sha256", "input_artifact_hash", "final_row_count", "row_offset", "max_final_rows", "target_mode", "target_fields", "target_fields_by_formulation_id", "source_table_ids", "evidence_source_mode", "stage2_evidence_blocks_path", "stage2_normalized_table_payloads_path", "source_text_path", "source_table_dir"]},
                "prompt_char_count": len(prompt_text),
                "prompt_status": "frozen_not_submitted",
            }
        ],
    )
    write_tsv(
        source_evidence_audit,
        ["paper_key", "evidence_source_mode", "block_id", "block_type", "evidence_kind", "source_type", "origin_locator", "table_id", "char_count", "removed_prompt_noise_lines", "included", "skip_reason"],
        evidence_audit_rows,
    )

    write_tsv(
        input_manifest,
        ["input_role", "path", "exists", "sha256_or_metadata"],
        [
            {"input_role": "final_table_tsv", "path": final_table_tsv, "exists": "yes", "sha256_or_metadata": sha256_file(final_table_tsv)},
            {"input_role": "decision_trace_tsv", "path": decision_trace_tsv, "exists": "yes", "sha256_or_metadata": sha256_file(decision_trace_tsv)},
            {"input_role": "scope_manifest_tsv", "path": scope_manifest_tsv, "exists": "yes", "sha256_or_metadata": sha256_file(scope_manifest_tsv)},
            {"input_role": "source_inventory_tsv", "path": source_inventory_tsv or "", "exists": "yes" if source_inventory_tsv else "not_provided", "sha256_or_metadata": sha256_file(source_inventory_tsv) if source_inventory_tsv else ""},
            {"input_role": "stage2_semantic_objects_dir", "path": semantic_objects_dir or "", "exists": "yes" if semantic_objects_dir and semantic_objects_dir.exists() else "not_provided", "sha256_or_metadata": "directory" if semantic_objects_dir else ""},
            {"input_role": "stage2_evidence_blocks_path", "path": evidence_blocks_path or "", "exists": "yes" if evidence_blocks_path and evidence_blocks_path.exists() else "no", "sha256_or_metadata": sha256_file(evidence_blocks_path) if evidence_blocks_path and evidence_blocks_path.exists() else ""},
            {"input_role": "stage2_normalized_table_payloads_path", "path": normalized_payloads_path or "", "exists": "yes" if normalized_payloads_path and normalized_payloads_path.exists() else "no", "sha256_or_metadata": sha256_file(normalized_payloads_path) if normalized_payloads_path and normalized_payloads_path.exists() else ""},
            {"input_role": "source_text_path", "path": text_path or "", "exists": "yes" if text_path and text_path.exists() else "no", "sha256_or_metadata": sha256_file(text_path) if text_path and text_path.exists() else ""},
            {"input_role": "source_table_dir", "path": table_dir or "", "exists": "yes" if table_dir and table_dir.exists() else "no", "sha256_or_metadata": json.dumps(sorted(p.name for p in table_dir.glob('*.csv')), ensure_ascii=False) if table_dir and table_dir.exists() else ""},
            {"input_role": "source_inventory_row", "path": source_inventory_tsv or "", "exists": "yes" if paper_key in source_inventory else "no", "sha256_or_metadata": json.dumps(source_inventory.get(paper_key, {}), ensure_ascii=False, sort_keys=True)},
        ],
    )

    run_context.write_text(
        "\n".join(
            [
                "# RUN_CONTEXT",
                "",
                "## Entrypoint",
                f"- entrypoint: `{ENTRYPOINT}`",
                f"- stage_boundary: `{STAGE_BOUNDARY}`",
                f"- boundary_class: `{BOUNDARY_CLASS}`",
                "",
                "## Benchmark-valid status",
                f"- benchmark_valid_status: `{BENCHMARK_VALID_STATUS}`",
                "- benchmark_valid: `no`",
                "- reason: `S5-3a is prompt construction only; no live LLM/API call and no benchmark scoring`",
                "- scope_policy: `targets all remaining S5-3-responsible direct-value fields after S5-2; excludes S5-2 mechanical table-cell fields`",
                "",
                "## Exact inputs",
                f"- final_table_tsv: `{final_table_tsv}`",
                f"- decision_trace_tsv: `{decision_trace_tsv}`",
                f"- scope_manifest_tsv: `{scope_manifest_tsv}`",
                f"- source_inventory_tsv: `{source_inventory_tsv or ''}`",
                f"- stage2_semantic_objects_dir: `{semantic_objects_dir or ''}`",
                f"- stage2_evidence_blocks_path: `{evidence_blocks_path or ''}`",
                f"- stage2_normalized_table_payloads_path: `{normalized_payloads_path or ''}`",
                f"- evidence_source_mode: `{evidence_metadata.get('evidence_source_mode', '')}`",
                f"- evidence_metadata: `{json.dumps(evidence_metadata, ensure_ascii=False, sort_keys=True)}`",
                f"- source_text_path: `{text_path or ''}`",
                f"- source_table_dir: `{table_dir or ''}`",
                f"- paper_key: `{paper_key}`",
                f"- row_offset: `{args.row_offset}`",
                f"- max_final_rows: `{args.max_final_rows}`",
                f"- target_mode: `{args.target_mode}`",
                f"- target_fields: `{json.dumps(target_fields, ensure_ascii=False)}`",
                f"- source_table_ids: `{json.dumps(source_table_ids_for_manifest, ensure_ascii=False)}`",
                "",
                "## Outputs",
                f"- prompts_jsonl: `{prompts_jsonl}`",
                f"- prompt_audit_tsv: `{prompt_audit}`",
                f"- input_manifest_tsv: `{input_manifest}`",
                f"- source_evidence_audit_tsv: `{source_evidence_audit}`",
                f"- prompt_template: `{prompt_template}`",
                f"- run_context_md: `{run_context}`",
                "",
                "## Stop rule",
                "- This run stops before S5-3b live LLM submission.",
                "- Raw LLM responses and candidate TSVs are not produced at this boundary.",
                "",
                "## Summary",
                f"- final_row_count: `{len(final_rows)}`",
                f"- prompt_count: `1`",
                f"- prompt_sha256: `{prompt_hash}`",
                f"- input_artifact_hash: `{input_artifact_hash}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "status": "success",
        "stage_boundary": STAGE_BOUNDARY,
        "out_dir": str(out_dir),
        "paper_key": paper_key,
        "final_row_count": len(final_rows),
        "target_mode": args.target_mode,
        "target_fields": target_fields,
        "target_fields_by_formulation_id": target_fields_by_formulation_id,
        "source_table_ids": source_table_ids_for_manifest,
        "evidence_source_mode": evidence_metadata.get("evidence_source_mode", ""),
        "evidence_metadata": evidence_metadata,
        "prompt_count": 1,
        "prompt_sha256": prompt_hash,
        "input_artifact_hash": input_artifact_hash,
        "outputs": {
            "prompts_jsonl": str(prompts_jsonl),
            "prompt_audit_tsv": str(prompt_audit),
            "input_manifest_tsv": str(input_manifest),
            "source_evidence_audit_tsv": str(source_evidence_audit),
            "prompt_template": str(prompt_template),
            "run_context_md": str(run_context),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
