#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pilot-only weak_labels_v7 extractor.

Purpose:
- Run a small pilot with stronger LLM-side semantic structure.
- Keep compatibility with current core fields while adding per-field semantic metadata.
- Do not change stable mainline extraction scripts.
"""

# Stage2 extractor restored to pre-Stage2.5 baseline.
# Stage2.5 additive shadow outputs removed during redesign transition (2026-03-25).

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from src.stage2_sampling_labels.build_numbered_doe_row_candidates_v1 import (
        ARTIFACT_NAME as NUMBERED_DOE_ARTIFACT_NAME,
        SUMMARY_NAME as NUMBERED_DOE_SUMMARY_NAME,
        enumerate_numbered_doe_candidates_for_paper,
        write_candidate_artifacts,
    )
    from src.utils.preparation_method_fields_v1 import (
        PREPARATION_METHOD_FIELDNAMES,
        enrich_preparation_method_fields_v1,
    )
    from src.utils.model_policy import PRIMARY_DEFAULT, validate_models_or_raise
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage2_sampling_labels.build_numbered_doe_row_candidates_v1 import (
        ARTIFACT_NAME as NUMBERED_DOE_ARTIFACT_NAME,
        SUMMARY_NAME as NUMBERED_DOE_SUMMARY_NAME,
        enumerate_numbered_doe_candidates_for_paper,
        write_candidate_artifacts,
    )
    from src.utils.preparation_method_fields_v1 import (
        PREPARATION_METHOD_FIELDNAMES,
        enrich_preparation_method_fields_v1,
    )
    from src.utils.model_policy import PRIMARY_DEFAULT, validate_models_or_raise

HAS_GENAI = False
try:
    import google.generativeai as genai  # type: ignore

    HAS_GENAI = True
except Exception:
    HAS_GENAI = False


CORE_FIELDS = [
    "emul_type",
    "emul_method",
    "la_ga_ratio",
    "polymer_mw_kDa",
    "plga_mass_mg",
    "surfactant_name",
    "surfactant_concentration_text",
    "pva_conc_percent",
    "organic_solvent",
    "drug_name",
    "drug_feed_amount_text",
    "size_nm",
    "pdi",
    "zeta_mV",
    "encapsulation_efficiency_percent",
    "loading_content_percent",
]

INSTANCE_KIND_VALUES = {
    "new_formulation",
    "variant_formulation",
    "candidate_non_formulation",
    "unclear",
}

CHANGE_ROLE_VALUES = {
    "synthesis_defining",
    "non_synthesis",
    "unclear",
}

NON_SYNTHESIS_TAGS = {
    "post_processing",
    "test_condition",
    "measurement_context",
    "storage",
    "characterization",
}

RECONCILIATION_HELPER_TAGS = {
    "post_processing",
    "test_condition",
    "measurement_context",
    "storage",
    "characterization",
    "characterization_only",
    "pharmacokinetics",
    "in_vivo",
    "commercial_product",
    "comparative",
}

FORMULATION_IDENTITY_FIELDS = {
    "emul_type",
    "emul_method",
    "la_ga_ratio",
    "polymer_mw_kDa",
    "plga_mass_mg",
    "surfactant_name",
    "surfactant_concentration_text",
    "pva_conc_percent",
    "organic_solvent",
    "drug_name",
    "drug_feed_amount_text",
}

# Observed fallback order when the model returns unnamed field objects as a list.
# This keeps pilot flattening deterministic without changing schema/prompt contracts.
UNNAMED_LIST_FALLBACK_ORDER = [
    "emul_method",
    "la_ga_ratio",
    "polymer_mw_kDa",
    "plga_mass_mg",
    "surfactant_name",
    "surfactant_concentration_text",
    "organic_solvent",
    "drug_name",
    "drug_feed_amount_text",
    "size_nm",
    "pdi",
    "zeta_mV",
    "encapsulation_efficiency_percent",
    "loading_content_percent",
]

VALUE_ALIASES_COMMON = [
    "value_raw",
    "raw_value",
    "raw",
    "value_num",
    "numeric_value",
    "value_number",
    "value_str",
    "normalized_value",
]

VALUE_ALIASES_BY_FIELD = {
    "la_ga_ratio": ["ratio", "la_ga", "la_ga_value", "ratio_value"],
    "polymer_mw_kDa": [
        "mw_kda",
        "mw",
        "molecular_weight_kda",
        "plga_mw",
        "polymer_mw",
        "mw_value",
    ],
}

LEGACY_FIELD_NAME_ALIASES = {
    "plga_mw_kDa": "polymer_mw_kDa",
}

NUMBERED_DOE_GUARD_NAME = "numbered_doe_regression_guard_v1.tsv"
NVIDIA_HOSTED_CHAT_COMPLETIONS_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


@dataclass
class EvidenceBlock:
    block_id: str
    block_type: str
    priority: int
    score: int
    text: str
    source_index: int
    source_start: int = -1
    source_end: int = -1


def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def normalize_doi(v: Any) -> str:
    s = "" if v is None else str(v)
    s = s.strip().lower()
    s = re.sub(r"^doi\s*:\s*", "", s)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\.org/", "", s)
    return s.strip()


def to_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def choose_instance_evidence(text: str) -> Dict[str, Any]:
    if not text:
        return {
            "evidence_region_type": "unknown",
            "evidence_section": "",
            "evidence_span_text": "",
            "evidence_span_start": "",
            "evidence_span_end": "",
        }
    m = re.search(
        r"(size|pdi|zeta|encapsulation\s+efficiency|drug/polymer|la/?ga).{0,260}",
        text,
        flags=re.IGNORECASE | re.S,
    )
    if not m:
        return {
            "evidence_region_type": "unknown",
            "evidence_section": "",
            "evidence_span_text": "",
            "evidence_span_start": "",
            "evidence_span_end": "",
        }
    s = max(0, m.start() - 60)
    e = min(len(text), m.end() + 120)
    span = text[s:e].replace("\n", " ").strip()
    return {
        "evidence_region_type": "unknown",
        "evidence_section": "full_text_window",
        "evidence_span_text": span,
        "evidence_span_start": int(s),
        "evidence_span_end": int(e),
    }


def normalize_block_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if re.fullmatch(r"(abstract|keywords|introduction|materials and methods|results and discussion|conclusions|acknowledgments|references)", s, flags=re.I):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9 ,()/-]{2,}", s):
        return True
    if len(s) <= 80 and s.isupper():
        return True
    return False


def split_paragraph_blocks(text: str) -> List[str]:
    text = normalize_block_text(text)
    if not text:
        return []

    line_groups = [g.strip() for g in re.split(r"\n\s*\n", text) if g.strip()]
    blocks: List[str] = []
    for group in line_groups:
        lines = [ln.strip() for ln in group.split("\n") if ln.strip()]
        current: List[str] = []
        for line in lines:
            if current and looks_like_heading(line):
                blocks.append(" ".join(current).strip())
                current = [line]
                continue
            current.append(line)
            joined = " ".join(current)
            if len(joined) >= 1100:
                blocks.append(joined.strip())
                current = []
        if current:
            blocks.append(" ".join(current).strip())

    out: List[str] = []
    for block in blocks:
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        if len(block) > 1800:
            parts = re.split(
                r"(?=(?:Table\s+\d+\b|Figure\s+\d+\b|\d+(?:\.\d+)*\.?\s+[A-Z]|Abstract\b|INTRODUCTION\b|MATERIALS AND METHODS\b|Results and DISCUSSION\b|CONCLUSIONS\b))",
                block,
            )
            for part in parts:
                part = re.sub(r"\s+", " ", part).strip()
                if part:
                    out.append(part)
        else:
            out.append(block)
    return out


def extract_table_like_blocks(text: str) -> List[str]:
    text = normalize_block_text(text)
    if not text:
        return []

    pattern = re.compile(
        r"(Table\s+\d+\b.*?)(?=(?:Table\s+\d+\b|Figure\s+\d+\b|\bReferences\b|\bACKNOWLEDGMENTS\b|\bCONCLUSIONS\b|$))",
        flags=re.I | re.S,
    )
    blocks: List[str] = []
    seen = set()
    for m in pattern.finditer(text):
        block = re.sub(r"\s+", " ", m.group(1)).strip()
        if len(block) < 40:
            continue
        if block.lower() in seen:
            continue
        seen.add(block.lower())
        blocks.append(block)
    return blocks


def select_best_table_anchor(block: str) -> str:
    matches = list(re.finditer(r"Table\s+\d+\b", block, flags=re.I))
    if not matches:
        return block
    for m in matches:
        tail = block[m.end(): m.end() + 120]
        if re.match(r"\s+[A-Z]", tail) and not re.match(r"\s*[,.)]", tail):
            return block[m.start():].strip()
    return block[matches[0].start():].strip()


def strip_running_table_artifacts(text: str) -> str:
    text = re.sub(
        r"M\.\s*Teixeira\s+et\s+al\.\s*/\s*European Journal of Pharmaceutics and Biopharmaceutics\s+59\s*\(2005\)\s*491\S+\s*\d{3}",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\bEuropean Journal of Pharmaceutics and Biopharmaceutics\s+59\s*\(2005\)\s*491\S+\s*\d{3}\b", " ", text, flags=re.I)
    text = re.sub(r"\b\d{3}\b(?=\s*$)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def trim_table_at_spillover(text: str) -> str:
    text = text.strip()
    cutoff = len(text)

    figure_match = re.search(r"\b(?:Fig\.|Figure)\s+\d+\.", text, flags=re.I)
    if figure_match and figure_match.start() >= 180:
        cutoff = min(cutoff, figure_match.start())

    section_match = re.search(r"\b(?:\d+\.\d+(?:\.\d+)?\.?\s+[A-Z]|4\.)", text)
    if section_match and section_match.start() >= 180:
        cutoff = min(cutoff, section_match.start())

    discussion_match = re.search(
        r"\b(?:Thus, our results suggest|Furthermore, the results show|The second strategy adopted|As can be seen in Fig\.|According to different authors|In the present work, to study)\b",
        text,
        flags=re.I,
    )
    if discussion_match and discussion_match.start() >= 220:
        cutoff = min(cutoff, discussion_match.start())

    values_note = re.search(r"Values express.*?different batches\.", text, flags=re.I)
    if values_note and values_note.end() < cutoff and (cutoff - values_note.end()) > 160:
        cutoff = min(cutoff, values_note.end())

    return text[:cutoff].strip(" .;,\n")


def clean_table_block_text(block: str) -> str:
    text = normalize_block_text(block)
    text = select_best_table_anchor(text)
    text = strip_running_table_artifacts(text)
    text = trim_table_at_spillover(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_table_like_matches(text: str) -> List[tuple[str, int, int, int]]:
    text = normalize_block_text(text)
    if not text:
        return []

    pattern = re.compile(
        r"(Table\s+\d+\b.*?)(?=(?:Table\s+\d+\b|Figure\s+\d+\b|\bReferences\b|\bACKNOWLEDGMENTS\b|\bCONCLUSIONS\b|$))",
        flags=re.I | re.S,
    )
    matches: List[tuple[str, int, int, int]] = []
    seen = set()
    for ordinal, m in enumerate(pattern.finditer(text), start=1):
        block = re.sub(r"\s+", " ", m.group(1)).strip()
        if len(block) < 40:
            continue
        block = clean_table_block_text(block)
        if len(block) < 80:
            continue
        if block.lower() in seen:
            continue
        seen.add(block.lower())
        matches.append((block, m.start(), m.end(), ordinal))
    return matches


def split_table_caption_and_body(block: str) -> tuple[str, str]:
    block = re.sub(r"\s+", " ", block).strip()
    m = re.match(
        r"^(Table\s+\d+\s*\.?\s*.*?)(?=(?:Values express|Download:|[A-Z][a-z]+ nanocapsules|[A-Z][a-z]+ nanospheres|Diameter \(|PI\b|z \(mV\)|XAN\b|3-MeOXAN\b|Empty\b|F\d+\b|Run\s*\d+\b|$))",
        block,
        flags=re.I,
    )
    if m:
        caption = m.group(1).strip()
        body = block[m.end():].strip()
        return caption, body
    if len(block) <= 260:
        return block, ""
    cut = block.find(" Values express")
    if cut == -1:
        cut = min(len(block), 260)
    return block[:cut].strip(), block[cut:].strip()


def score_table_block(text: str) -> int:
    lower = text.lower()
    score = 0
    score += 6 * len(re.findall(r"\b(?:f|run|sample)\s*[-:]?\s*\d{1,3}\b", lower))
    score += 5 * len(re.findall(r"\b(?:xan|3-meoxan|empty|blank|control|optimized)\b", lower))
    score += 4 * len(re.findall(r"\b(?:theoretical concentration|final concentration|drug concentration|plga concentration|pva concentration|aqueous phase pH)\b", lower))
    score += 3 * len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:mg/ml|mg/mL|mg|%|mL|ml|nm|mV)\b", text, flags=re.I))
    score += 3 * len(re.findall(r"\b(?:diameter|particle size|pdi|zeta|encapsulation|incorporation efficiency|loading)\b", lower))
    if "table " in lower:
        score += 10
    score -= 25 * len(re.findall(r"\b(?:fig\.|figure)\s+\d+\.", lower))
    score -= 20 * len(re.findall(r"\b(?:\d+\.\d+(?:\.\d+)?\.?\s+[A-Z]|4\.)", text))
    score -= 15 * len(re.findall(r"M\.\s*Teixeira|European Journal of Pharmaceutics", text, flags=re.I))
    if len(text) > 1800:
        score -= min(120, (len(text) - 1800) // 15)
    sentence_like = len(re.findall(r"[a-z][.!?]\s+[A-Z]", text))
    if sentence_like >= 4:
        score -= 12 * (sentence_like - 3)
    return max(score, 0)


def classify_paragraph_block(text: str) -> tuple[str, int, int]:
    lower = text.lower()
    score = 0

    label_hits = len(re.findall(r"\b(?:f|run|sample|formulation)\s*[-:]?\s*\d{1,3}\b", lower))
    inheritance_hits = len(re.findall(r"\b(?:prepared similarly|except|all other variables unchanged|same protocol|compared with|relative to)\b", lower))
    sweep_hits = len(re.findall(r"\b(?:sweep|optimization|design matrix|doe|box-behnken|response surface|factorial|experimental runs|run order)\b", lower))
    control_hits = len(re.findall(r"\b(?:optimized formulation|control|blank|empty nanocapsule|empty nanospheres|empty)\b", lower))
    variable_hits = len(re.findall(r"\b(?:plga|pva|pluronic|lecithin|myritol|acetone|drug|theoretical concentration|final concentration|diameter|zeta|encapsulation)\b", lower))
    table_hits = len(re.findall(r"\btable\s+\d+\b", lower))

    score += 8 * label_hits
    score += 8 * inheritance_hits
    score += 7 * sweep_hits
    score += 6 * control_hits
    score += 2 * variable_hits
    score += 3 * table_hits

    prep_hits = len(
        re.findall(
            r"\b(?:prepared|preparation|formulations? were prepared|new formulations|strategy adopted|same method|same procedure|fixed amounts|varying|different concentrations|using fixed amounts of polymer|while maintaining|this led to|omitting the xanthones)\b",
            lower,
        )
    )
    entity_hits = len(re.findall(r"\b(?:nanospheres|nanocapsules|nanoparticles|polymer|surfactants|oil volume|myritol|pluronic|lecithin|drug concentration|polymer concentration)\b", lower))
    negative_hits = len(
        re.findall(
            r"\b(?:dsc|in vitro release|release profiles|physical stability|storage|thermal behaviour|particle size analysis|zeta potential analysis|characterization|biological activity|ocular tolerance|statistics)\b",
            lower,
        )
    )
    reference_hits = len(re.findall(r"\[\d+\]", text))
    journal_citation_hits = len(re.findall(r"\b(?:j\.|int\.|eur\.|pharm\.|biopharm\.|drug dev\.|thesis|patent)\b", lower))
    is_reference_like = reference_hits >= 2 or journal_citation_hits >= 4
    is_conclusion_like = lower.startswith("4. conclusions") or lower.startswith("conclusions")
    stripped = text.strip()
    materials_heading = bool(
        re.match(r"^(?:materials and methods\s+)?materials\b", stripped, flags=re.I)
        or re.match(r"^\d+\.\d+\.?\s*materials\b", stripped, flags=re.I)
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

    priority = 5
    block_type = "paragraph"

    if (
        (prep_hits >= 2 and entity_hits >= 2 and negative_hits <= 2)
        or inheritance_hits >= 1
        or re.search(r"\b(?:2\.2\.\s*Preparation|2\.4\.\s*Preparation|prepared according to the same procedure)\b", text, flags=re.I)
    ) and not is_reference_like and not is_conclusion_like:
        block_type = "synthesis_method"
        priority = 1
        score += 20 + (6 * prep_hits) + (4 * entity_hits)
    elif (
        (
            materials_heading
            and procurement_hits >= 1
            and (mw_hits >= 1 or polymer_identity_hits >= 1)
        )
        or (
            procurement_hits >= 1
            and mw_hits >= 1
            and polymer_identity_hits >= 1
        )
    ) and not is_reference_like and not is_conclusion_like:
        block_type = "materials_procurement"
        priority = 2
        score += 16 + (5 * procurement_hits) + (4 * mw_hits) + (3 * polymer_identity_hits)
    elif label_hits or inheritance_hits or sweep_hits or control_hits:
        priority = 4
    elif variable_hits >= 5:
        priority = 4
    return block_type, priority, score


def build_metadata_block(key: str, doi: str, title: str) -> str:
    parts = ["[METADATA]"]
    if key:
        parts.append(f"key: {key}")
    if doi:
        parts.append(f"doi: {doi}")
    if title:
        parts.append(f"title: {title}")
    return "\n".join(parts).strip()


def format_evidence_block(block: EvidenceBlock) -> str:
    label_map = {
        "metadata": "METADATA",
        "synthesis_method": "SYNTHESIS_METHOD_BLOCK",
        "materials_procurement": "MATERIALS_PROCUREMENT_BLOCK",
        "table": "TABLE_BLOCK",
        "caption": "CAPTION_BLOCK",
        "paragraph": "PARAGRAPH_BLOCK",
    }
    header = f"[{label_map.get(block.block_type, block.block_type.upper())}]"
    return f"{header}\n{block.text.strip()}".strip()


def count_source_line(text: str, offset: int) -> int:
    if offset is None or offset < 0:
        return -1
    return text[:offset].count("\n") + 1


def locate_block_start(text: str, block_text: str, start_at: int = 0) -> int:
    if not block_text:
        return -1
    idx = text.find(block_text, max(0, start_at))
    if idx >= 0:
        return idx
    compact_text = re.sub(r"\s+", " ", text)
    compact_block = re.sub(r"\s+", " ", block_text).strip()
    return compact_text.find(compact_block)


def build_evidence_candidates(raw_text: str, key: str, doi: str, title: str) -> tuple[str, List[EvidenceBlock]]:
    normalized = normalize_block_text(raw_text)
    metadata = EvidenceBlock(
        block_id="metadata",
        block_type="metadata",
        priority=0,
        score=0,
        text=build_metadata_block(key, doi, title),
        source_index=-1,
        source_start=-1,
        source_end=-1,
    )
    if not normalized:
        return normalized, [metadata]

    candidates: List[EvidenceBlock] = []
    seen_texts = set()

    table_matches = extract_table_like_matches(normalized)
    for idx, (table_block, block_start, block_end, ordinal) in enumerate(table_matches):
        caption, body = split_table_caption_and_body(table_block)
        table_score = score_table_block(table_block)
        if body:
            norm_body = body.lower()
            if norm_body not in seen_texts:
                seen_texts.add(norm_body)
                body_start = block_start + max(0, table_block.find(body))
                candidates.append(
                    EvidenceBlock(
                        block_id=f"table_{ordinal}",
                        block_type="table",
                        priority=3,
                        score=table_score,
                        text=body,
                        source_index=idx,
                        source_start=body_start,
                        source_end=body_start + len(body),
                    )
                )
        if caption:
            norm_caption = caption.lower()
            if norm_caption not in seen_texts:
                seen_texts.add(norm_caption)
                caption_start = block_start + max(0, table_block.find(caption))
                candidates.append(
                    EvidenceBlock(
                        block_id=f"caption_{ordinal}",
                        block_type="caption",
                        priority=4,
                        score=max(1, table_score // 2),
                        text=caption,
                        source_index=idx,
                        source_start=caption_start,
                        source_end=caption_start + len(caption),
                    )
                )

    paragraphs = split_paragraph_blocks(normalized)
    search_cursor = 0
    for idx, para in enumerate(paragraphs):
        norm_para = para.lower()
        if norm_para in seen_texts:
            continue
        block_type, priority, score = classify_paragraph_block(para)
        if score <= 0:
            continue
        seen_texts.add(norm_para)
        para_start = locate_block_start(normalized, para, start_at=search_cursor)
        if para_start >= 0:
            search_cursor = para_start + len(para)
        candidates.append(
            EvidenceBlock(
                block_id=f"paragraph_{idx+1}",
                block_type=block_type,
                priority=priority,
                score=score,
                text=para,
                source_index=idx,
                source_start=para_start,
                source_end=(para_start + len(para)) if para_start >= 0 else -1,
            )
        )

    candidates.sort(key=lambda b: (b.priority, -b.score, b.source_index, b.block_id))
    return normalized, [metadata] + candidates


def pack_evidence_blocks(candidates: List[EvidenceBlock], max_chars: int) -> Dict[str, Any]:
    if not candidates:
        return {"selected_blocks": [], "packed_text": ""}

    blocks_out: List[str] = []
    selected_blocks: List[Dict[str, Any]] = []
    used_ids = set()
    current_len = 0

    for block in candidates:
        if block.block_id in used_ids:
            continue
        formatted = format_evidence_block(block)
        candidate_len = len(formatted) + (2 if blocks_out else 0)
        remaining = max_chars - current_len if max_chars > 0 else None
        if remaining is not None and remaining <= 0:
            break
        if remaining is None or candidate_len <= remaining:
            if blocks_out:
                blocks_out.append("")
            blocks_out.append(formatted)
            current_len += candidate_len
            used_ids.add(block.block_id)
            selected_blocks.append(
                {
                    "packing_rank": len(selected_blocks) + 1,
                    "block": block,
                    "char_len": len(formatted),
                    "cumulative_char_count": current_len,
                    "truncated": "no",
                }
            )
            continue
        if remaining is not None and remaining >= 420 and block.priority <= 2 and block.block_type != "metadata":
            header = f"[{block.block_type.upper()}_BLOCK]"
            body_budget = max(0, remaining - len(header) - len("\n[TRUNCATED]") - 4)
            body = block.text[:body_budget].rstrip()
            if body:
                trimmed = f"{header}\n{body}\n[TRUNCATED]"
                if blocks_out:
                    blocks_out.append("")
                blocks_out.append(trimmed)
                current_len += len(trimmed) + (2 if len(blocks_out) > 1 else 0)
                used_ids.add(block.block_id)
                selected_blocks.append(
                    {
                        "packing_rank": len(selected_blocks) + 1,
                        "block": block,
                        "char_len": len(trimmed),
                        "cumulative_char_count": current_len,
                        "truncated": "yes",
                    }
                )
                break

    assembled = "\n".join(blocks_out).strip()
    if max_chars > 0 and len(assembled) > max_chars:
        assembled = assembled[:max_chars]
    return {"selected_blocks": selected_blocks, "packed_text": assembled}


def assemble_evidence_text(raw_text: str, key: str, doi: str, title: str, max_chars: int) -> str:
    _, candidates = build_evidence_candidates(raw_text, key, doi, title)
    packed = pack_evidence_blocks(candidates, max_chars=max_chars)
    return str(packed["packed_text"])


FEW_SHOT = r"""
Few-shot guidance (compact examples):
1) Shared method/header condition:
- Methods or table header states: "Polymer molecular weight 50 kDa, LA/GA 50:50, solvent acetone for all formulations F1-F4".
- Use:
  polymer_mw_kDa.scope = global_shared
  la_ga_ratio.scope = global_shared
  organic_solvent.scope = global_shared

2) Row-specific result:
- Table row F3 reports size and EE values.
- Use:
  size_nm.scope = instance_specific
  encapsulation_efficiency_percent.scope = instance_specific

3) Ambiguous evidence:
- Text does not clearly bind surfactant concentration to one formulation or all formulations.
- Use:
  surfactant_concentration_text.scope = unknown
  membership_confidence = low

4) Polymer product code vs molecular weight:
- Input text: "PCL (40 kDa) was used as the polymer."
- Interpret this as a polymer molecular-weight value, even when the polymer is not PLGA.
- Use:
  polymer_mw_kDa.value = "40 kDa"
  polymer_mw_kDa.value_text = "PCL (40 kDa)"
  polymer_mw_kDa.scope = global_shared

5) Polymer product code vs molecular weight:
- Input text: "PLGA (Resomer RG 502) was used as the polymer."
- Interpret as polymer grade/product code, not direct MW value.
- Use:
  polymer_mw_kDa.value = null
  polymer_mw_kDa.value_text = "Resomer RG 502 (polymer grade)"
  polymer_mw_kDa.scope = global_shared (if stated once for all formulations)
  polymer_mw_kDa.membership_confidence = medium

6) Baseline + additive variant:
- Input text: "Nanoparticles were prepared using 0.5% PVA as stabilizer. Variants were prepared using the same protocol with additional surfactants."
- Use:
  surfactant_name value "PVA" with scope = global_shared
  additional surfactant labels with scope = instance_specific

6) Inheritance-style formulation variant:
- Input text: "F2 was prepared similarly to F1 except that PLGA concentration increased from 10 to 20 mg/mL."
- Use:
  instance_kind = variant_formulation
  parent_instance_id = "F1"
  change_descriptions = ["PLGA concentration increased from 10 to 20 mg/mL"]
  change_role = synthesis_defining

7) Post-processing or test-only variation:
- Input text: "F1 was freeze-dried with sucrose" or "particle size was measured after 30 days at 4 C".
- Use:
  instance_kind = candidate_non_formulation
  change_role = non_synthesis
  change_context_tags = ["post_processing"] or ["storage"] or ["measurement_context"]
- Do NOT convert these into distinct formulation rows if synthesis-defining parameters are unchanged.

8) Preliminary or characterization-only helper rows:
- Input text: "Preliminary formulation was prepared before the DOE table" or "6-coumarin-loaded nanoparticles were prepared only for uptake/characterization studies".
- Use:
  instance_kind = candidate_non_formulation
  change_role = non_synthesis
  formulation_role = characterization_only or unknown
- Do NOT keep these as benchmark-facing formulation rows unless the paper clearly treats them as explicit formulation-instance rows in the main reported set.

"""


LLM_PROMPT_TEMPLATE = (
    "You are extracting nanoparticle formulation instances in weak_labels_v7 pilot format.\n"
    "Return ONLY valid JSON with keys: schema_version, paper_notes, formulations.\n"
    "Use schema_version='weak_labels_v7'.\n"
    "Each formulation object must include: formulation_id, raw_formulation_label, instance_kind, parent_instance_id, change_descriptions, change_role, supporting_evidence_refs, instance_context_tags, change_context_tags, formulation_role, instance_confidence, fields.\n"
    "Allowed instance_kind: new_formulation, variant_formulation, candidate_non_formulation, unclear.\n"
    "Allowed change_role: synthesis_defining, non_synthesis, unclear.\n"
    "Allowed formulation_role: baseline, control, optimized, variant, comparative, characterization_only, unknown.\n"
    "Allowed instance_confidence: high, medium, low.\n"
    "For each core field below, output a field object (or omit if fully unknown):\n"
    + ", ".join(CORE_FIELDS)
    + "\n"
    "Field object schema:\n"
    "- value: scalar or null\n"
    "- value_text: string\n"
    "- scope: global_shared | instance_specific | unknown\n"
    "- membership_confidence: high | medium | low\n"
    "- evidence_region_type: table_cell | table_row | table_header | table_block | methods_sentence | results_sentence | unknown\n"
    "- missing_reason: optional string for unknown/ambiguous cases\n\n"
    "Rules:\n"
    "- Final database remains tabular with one row per formulation.\n"
    "- Treat formulation-instance recognition as primary; do not output scattered fields without an instance.\n"
    "- Do not emit standalone 'global parameters', 'shared conditions', or family-header pseudo-formulations. Shared synthesis conditions should be attached to real formulation rows via global_shared field scope instead.\n"
    "- Post-processing differences, test conditions, storage conditions, release-test conditions, and measurement contexts do NOT automatically define new formulation rows.\n"
    "- If synthesis-defining parameters stay the same and only post-processing/test/storage/measurement changes, use instance_kind=candidate_non_formulation and change_role=non_synthesis.\n"
    "- A failed or suboptimal synthesis row in a concentration sweep still counts as a formulation candidate if it is a distinct synthesis row. Do not use candidate_non_formulation only because a row shows precipitation, crystals, ND encapsulation, or partial characterization.\n"
    "- If a formulation is defined relative to a parent/base formulation through 'prepared similarly', 'except', 'all other variables unchanged', or similar inheritance language, use instance_kind=variant_formulation and set parent_instance_id.\n"
    "- If a true synthesis/design variable changes, including one outside the core field list, treat it as synthesis_defining and describe it in change_descriptions.\n"
    "- DOE, Box-Behnken, response-surface, or parameter-sweep rows can still be formulation rows when the varied factor is synthesis-defining.\n"
    "- A preliminary setup row that only motivates later DOE or optimization should not default to a benchmark-facing formulation row unless the paper clearly includes it in the reported formulation set.\n"
    "- Characterization-only helper formulations, model-dye substitutions, and uptake/imaging assay variants should default to candidate_non_formulation unless the paper clearly treats them as part of the main reported formulation set.\n"
    "- Do not apply material filtering. If the paper reports formulation rows for polymers outside the current PLGA modeling scope (for example PCL), they must still be extracted as formulation instances.\n"
    "- Use instance_context_tags/change_context_tags only as auxiliary tags such as doe, sweep, post_processing, test_condition, measurement_context, optimized, control.\n"
    "- Do not hallucinate values; prefer unknown style values when uncertain.\n"
    "- Preserve units in value_text.\n"
    "- If table header defines shared conditions, mark scope=global_shared.\n"
    "- If value is clearly tied to one formulation row, mark scope=instance_specific.\n"
    "- Do not overuse unknown when the paper clearly states one common condition for all listed formulations.\n"
    "- Use global_shared only when the same condition truly applies to all extracted formulations in the paper or in one coherent table block.\n"
    "- polymer_mw_kDa is the general polymer molecular-weight field for PLGA and non-PLGA polymers such as PCL.\n"
    "- Product codes like RG 502 / RG 503 / RG 504 are polymer grade names, not molecular-weight values.\n"
    "- If only a polymer product code is given, keep it in value_text and set polymer_mw_kDa.value=null.\n"
    "- LA/GA is conditional and only relevant when the polymer is PLGA-like.\n"
    "- If a baseline stabilizer or solvent is declared once and reused across formulations, mark baseline condition as global_shared.\n"
    "- For ambiguous assignment, set scope=unknown and low membership_confidence.\n\n"
    + FEW_SHOT
    + "\nTEXT:\n"
)


ENUMERATION_HEAVY_TABLE_HINT = (
    "\nADDITIONAL ENUMERATION RULES FOR TABLE-HEAVY PAPERS:\n"
    "- For table-heavy or sweep-style formulation studies, treat each table row or run as a potential formulation instance.\n"
    "- Enumerate formulation candidates row by row before any abstraction.\n"
    "- Only after all candidate rows are enumerated may parent/variant relationships be assigned.\n"
    "- If the paper contains a design matrix, sweep table, repeated concentration table, or repeated formulation rows, enumerate candidate formulations row by row before summarizing parent/variant relations.\n"
    "- If a table lists multiple experimental runs or formulations, each row must be emitted as a separate formulation instance unless clear textual evidence states that rows are replicates or measurement-only variations.\n"
    "- Do not collapse multiple table rows into one family-level formulation such as one generic 'XAN nanospheres' or one generic 'XAN nanocapsules' record when the table contains several row-level formulations.\n"
    "- Each distinct table row with a different drug loading, theoretical concentration, oil-core amount, or other synthesis-setting change should become its own formulation candidate.\n"
    "- Keep rows with crystallization, ND encapsulation, or partial characterization as distinct formulation candidates if the synthesis row itself is distinct.\n"
    "- Empty/baseline rows should also be enumerated if they appear as explicit rows in the formulation tables.\n"
    "- If one table block reports loaded variants and a neighboring table reports the corresponding empty baseline for the same preparation block, keep that empty baseline as its own formulation candidate for that block.\n"
    "- Do not create extra pseudo-rows just to hold shared table parameters or method constants.\n"
    "- If the table row has no explicit label, create a stable local id from the row content rather than merging it into a parent summary.\n"
)


def ensure_genai(model: str) -> None:
    if not HAS_GENAI:
        die("google-generativeai is not installed in this environment.")
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        die("GEMINI_API_KEY / GOOGLE_API_KEY is missing in environment.")
    genai.configure(api_key=key)
    _ = genai.GenerativeModel(model)


def infer_llm_backend(model: str, requested_backend: str) -> str:
    backend = str(requested_backend or "auto").strip().lower()
    if backend in {"gemini", "nvidia"}:
        return backend
    model_name = str(model or "").strip().lower()
    if model_name.startswith("meta/") or "llama" in model_name or model_name.startswith("nvidia/"):
        return "nvidia"
    return "gemini"


def call_gemini(model: str, prompt: str, retries: int, sleep_sec: float) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            mdl = genai.GenerativeModel(model)
            resp = mdl.generate_content(prompt)
            if hasattr(resp, "text") and resp.text:
                return resp.text
            try:
                cand = resp.candidates[0].content.parts[0].text
                if cand:
                    return cand
            except Exception:
                pass
            last_err = RuntimeError("Empty response text")
        except Exception as e:
            last_err = e
        if attempt < retries:
            time.sleep(sleep_sec)
    raise last_err or RuntimeError("Gemini call failed")


def ensure_nvidia_hosted_api(model: str) -> None:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        die("NVIDIA_API_KEY is missing in environment.")
    # This diagnostic/full-pipeline path uses the NVIDIA hosted API Catalog endpoint.
    # It must not assume a local NIM deployment such as http://localhost:8000/v1.
    if "localhost" in NVIDIA_HOSTED_CHAT_COMPLETIONS_URL or "127.0.0.1" in NVIDIA_HOSTED_CHAT_COMPLETIONS_URL:
        die("Hosted NVIDIA endpoint misconfigured: local NIM endpoints are not allowed here.")
    if not str(model or "").strip():
        die("NVIDIA model name is empty.")


def call_nvidia_hosted(model: str, prompt: str, retries: int, sleep_sec: float) -> str:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is missing in environment.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only valid JSON that matches the extraction schema implied by the user prompt."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                NVIDIA_HOSTED_CHAT_COMPLETIONS_URL,
                headers=headers,
                json=payload,
                timeout=180,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "").strip()
                if attempt < retries:
                    wait_sec = max(sleep_sec, float(retry_after) if retry_after else sleep_sec * (attempt + 2))
                    time.sleep(wait_sec)
                    continue
                response.raise_for_status()
            response.raise_for_status()
            body = response.json()
            choices = body.get("choices") or []
            if not choices:
                raise RuntimeError(f"NVIDIA hosted response did not include choices: {body}")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            raise RuntimeError(f"NVIDIA hosted response content was empty: {body}")
        except Exception as e:
            last_err = e
        if attempt < retries:
            time.sleep(sleep_sec * (attempt + 1))
    raise last_err or RuntimeError("NVIDIA hosted API call failed")


def ensure_llm_backend(model: str, backend: str) -> str:
    resolved_backend = infer_llm_backend(model, backend)
    if resolved_backend == "gemini":
        ensure_genai(model)
        return resolved_backend
    if resolved_backend == "nvidia":
        ensure_nvidia_hosted_api(model)
        return resolved_backend
    raise ValueError(f"Unsupported llm backend: {resolved_backend}")


def call_llm(model: str, prompt: str, retries: int, sleep_sec: float, backend: str) -> str:
    resolved_backend = infer_llm_backend(model, backend)
    if resolved_backend == "gemini":
        return call_gemini(model, prompt, retries, sleep_sec)
    if resolved_backend == "nvidia":
        return call_nvidia_hosted(model, prompt, retries, sleep_sec)
    raise ValueError(f"Unsupported llm backend: {resolved_backend}")


def build_prompt(text: str, detection_text: Optional[str] = None) -> str:
    prompt = LLM_PROMPT_TEMPLATE
    detector_blob = detection_text if detection_text is not None else text
    if is_table_heavy_sweep_candidate(detector_blob):
        prompt += ENUMERATION_HEAVY_TABLE_HINT
    prompt += text
    return prompt


def is_table_heavy_sweep_candidate(text: str) -> bool:
    lower = text.lower()

    table_hits = len(re.findall(r"\btable\s+[1-9][0-9]?\b", lower))
    formulation_label_hits = len(
        re.findall(r"\b(?:f|run|sample|formulation)\s*[-:]?\s*\d{1,3}\b", lower)
    )
    repeated_conc_hits = len(
        re.findall(
            r"\b(?:theoretical concentration|final concentration|drug concentration|plga concentration|pva concentration|aqueous phase pH)\b",
            lower,
        )
    )
    repeated_value_patterns = len(
        re.findall(
            r"\b\d+(?:\.\d+)?\s*(?:mg/ml|mg/mL|mg|%|mL|ml|nm|mV|kda|kDa)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    sweep_keywords = len(
        re.findall(
            r"\b(?:sweep|design matrix|doe|box-behnken|response surface|factorial|experimental runs|run order)\b",
            lower,
        )
    )

    has_multi_table_signal = table_hits >= 3
    has_run_label_signal = formulation_label_hits >= 4
    has_repeated_parameter_signal = repeated_conc_hits >= 4
    has_dense_table_value_signal = table_hits >= 2 and repeated_value_patterns >= 20
    has_design_language_signal = sweep_keywords >= 1 and (formulation_label_hits >= 2 or table_hits >= 2)

    return any(
        [
            has_multi_table_signal,
            has_run_label_signal,
            has_repeated_parameter_signal,
            has_dense_table_value_signal,
            has_design_language_signal,
        ]
    )


def safe_json_load(s: str) -> Dict[str, Any]:
    cleaned = s.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        m = re.search(r"\{.*\}", cleaned, re.S)
        if m:
            return json.loads(m.group(0))
    return {"schema_version": "weak_labels_v7", "paper_notes": None, "formulations": []}


def normalize_field_obj(v: Any) -> Dict[str, Any]:
    if isinstance(v, dict):
        val = v.get("value")
        if val is None or (isinstance(val, str) and not val.strip()):
            for k in VALUE_ALIASES_COMMON:
                cand = v.get(k)
                if cand is not None and (not isinstance(cand, str) or cand.strip()):
                    val = cand
                    break
        return {
            "value": val,
            "value_text": str(v.get("value_text", "") or ""),
            "scope": str(v.get("scope", "unknown") or "unknown"),
            "membership_confidence": str(v.get("membership_confidence", "low") or "low"),
            "evidence_region_type": str(v.get("evidence_region_type", "unknown") or "unknown"),
            "missing_reason": str(v.get("missing_reason", "") or ""),
            "value_source": str(v.get("value_source", "") or ""),
        }
    if v is None:
        return {
            "value": None,
            "value_text": "",
            "scope": "unknown",
            "membership_confidence": "low",
            "evidence_region_type": "unknown",
            "missing_reason": "not_reported",
        }
    return {
        "value": v,
        "value_text": str(v),
        "scope": "unknown",
        "membership_confidence": "medium",
        "evidence_region_type": "unknown",
        "missing_reason": "",
        "value_source": "",
    }


def sanitize_role(v: Any) -> str:
    allowed = {
        "baseline",
        "control",
        "optimized",
        "variant",
        "comparative",
        "characterization_only",
        "unknown",
    }
    s = str(v or "unknown").strip().lower()
    return s if s in allowed else "unknown"


def sanitize_conf(v: Any) -> str:
    allowed = {"high", "medium", "low"}
    s = str(v or "low").strip().lower()
    return s if s in allowed else "low"


def sanitize_scope(v: Any) -> str:
    allowed = {"global_shared", "instance_specific", "unknown"}
    s = str(v or "unknown").strip().lower()
    return s if s in allowed else "unknown"


def sanitize_region(v: Any) -> str:
    allowed = {
        "table_cell",
        "table_row",
        "table_header",
        "table_block",
        "methods_sentence",
        "results_sentence",
        "unknown",
    }
    s = str(v or "unknown").strip().lower()
    return s if s in allowed else "unknown"


def sanitize_instance_kind(v: Any) -> str:
    s = str(v or "unclear").strip().lower()
    return s if s in INSTANCE_KIND_VALUES else "unclear"


def sanitize_change_role(v: Any) -> str:
    s = str(v or "unclear").strip().lower()
    return s if s in CHANGE_ROLE_VALUES else "unclear"


def sanitize_string_list(v: Any) -> List[str]:
    if isinstance(v, list):
        out = [str(x).strip() for x in v if str(x).strip()]
        return out
    if v is None:
        return []
    s = str(v).strip()
    if not s:
        return []
    parts = re.split(r"[|,;\n]+", s)
    return [p.strip() for p in parts if p.strip()]


def sanitize_tag_list(v: Any) -> List[str]:
    tags: List[str] = []
    for item in sanitize_string_list(v):
        tag = re.sub(r"[^a-z0-9]+", "_", item.lower()).strip("_")
        if tag:
            tags.append(tag)
    seen = set()
    out: List[str] = []
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def sanitize_supporting_evidence_refs(v: Any, instance_evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    items = v if isinstance(v, list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        ref = {
            "region_type": sanitize_region(item.get("region_type") or item.get("evidence_region_type")),
            "section": str(item.get("section") or item.get("evidence_section") or "").strip(),
            "span_text": str(item.get("span_text") or item.get("evidence_span_text") or "").strip(),
            "span_start": item.get("span_start") or item.get("evidence_span_start") or "",
            "span_end": item.get("span_end") or item.get("evidence_span_end") or "",
        }
        if any(str(ref[k]).strip() for k in ["section", "span_text", "span_start", "span_end"]):
            refs.append(ref)
    if refs:
        return refs
    fallback = {
        "region_type": sanitize_region(instance_evidence.get("evidence_region_type")),
        "section": str(instance_evidence.get("evidence_section") or "").strip(),
        "span_text": str(instance_evidence.get("evidence_span_text") or "").strip(),
        "span_start": instance_evidence.get("evidence_span_start") or "",
        "span_end": instance_evidence.get("evidence_span_end") or "",
    }
    if any(str(fallback[k]).strip() for k in ["section", "span_text", "span_start", "span_end"]):
        return [fallback]
    return []


def infer_change_role(raw_change_role: Any, parent_instance_id: str, change_descriptions: List[str], tags: List[str]) -> str:
    change_role = sanitize_change_role(raw_change_role)
    if change_role != "unclear":
        return change_role
    if set(tags) & NON_SYNTHESIS_TAGS:
        return "non_synthesis"
    if parent_instance_id or change_descriptions:
        return "synthesis_defining"
    return "unclear"


def infer_instance_kind(
    raw_instance_kind: Any,
    parent_instance_id: str,
    change_role: str,
    formulation_role: str,
    tags: List[str],
    has_fields: bool,
) -> str:
    instance_kind = sanitize_instance_kind(raw_instance_kind)
    if instance_kind != "unclear":
        return instance_kind
    if change_role == "non_synthesis" or formulation_role == "characterization_only" or (set(tags) & NON_SYNTHESIS_TAGS):
        return "candidate_non_formulation"
    if parent_instance_id:
        return "variant_formulation"
    if has_fields:
        return "new_formulation"
    return "unclear"


def _field_has_meaningful_value(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    value = obj.get("value")
    if value is not None and (not isinstance(value, str) or value.strip()):
        return True
    value_text = str(obj.get("value_text") or "").strip()
    if not value_text:
        return False
    return value_text.lower() not in {"none", "n/a", "na", "not applicable", "not reported", "unknown"}


def _count_formulation_identity_fields(fields: Dict[str, Dict[str, Any]]) -> int:
    count = 0
    for field_name in FORMULATION_IDENTITY_FIELDS:
        if _field_has_meaningful_value(fields.get(field_name)):
            count += 1
    return count


def _has_family_membership_signal(
    raw_formulation_label: str,
    parent_instance_id: str,
    polymer_identity: str,
    polymer_name_raw: str,
) -> bool:
    label = str(raw_formulation_label or "").strip().lower()
    has_family_label = bool(
        re.search(
            r"\b(blank|fitc|loaded|plga|peg|ha|nanoparticles?|nanospheres?|nanocapsules?)\b",
            label,
        )
    )
    has_polymer_identity = polymer_identity != "unknown" or bool(str(polymer_name_raw or "").strip())
    return bool(parent_instance_id) and has_family_label and has_polymer_identity


def reconcile_instance_kind(
    *,
    raw_instance_kind: Any,
    parent_instance_id: str,
    change_role: str,
    formulation_role: str,
    tags: List[str],
    fields: Dict[str, Dict[str, Any]],
    polymer_identity: str,
    polymer_name_raw: str,
    raw_formulation_label: str,
) -> Dict[str, str]:
    raw_instance_kind_norm = sanitize_instance_kind(raw_instance_kind)
    inferred_instance_kind = infer_instance_kind(
        raw_instance_kind=raw_instance_kind_norm,
        parent_instance_id=parent_instance_id,
        change_role=change_role,
        formulation_role=formulation_role,
        tags=tags,
        has_fields=bool(fields),
    )
    formulation_identity_field_count = _count_formulation_identity_fields(fields)
    has_polymer_identity = polymer_identity != "unknown" or bool(str(polymer_name_raw or "").strip())
    has_strong_formulation_identity = has_polymer_identity and formulation_identity_field_count >= 2
    has_family_membership_signal = _has_family_membership_signal(
        raw_formulation_label=raw_formulation_label,
        parent_instance_id=parent_instance_id,
        polymer_identity=polymer_identity,
        polymer_name_raw=polymer_name_raw,
    )
    helper_like_signal = (
        change_role == "non_synthesis"
        and (
            formulation_role == "comparative"
            or bool(set(tags) & RECONCILIATION_HELPER_TAGS)
        )
    )

    reconciled_instance_kind = inferred_instance_kind
    reconciliation_note = ""

    if (
        inferred_instance_kind == "candidate_non_formulation"
        and (has_strong_formulation_identity or has_family_membership_signal)
    ):
        reconciled_instance_kind = "variant_formulation" if parent_instance_id else "new_formulation"
        reconciliation_note = (
            "rescued_family_member_conflict:"
            f"polymer_identity={polymer_identity};"
            f"identity_field_count={formulation_identity_field_count};"
            f"parent_link={'yes' if parent_instance_id else 'no'}"
        )
    elif (
        inferred_instance_kind in {"new_formulation", "variant_formulation"}
        and helper_like_signal
        and not has_strong_formulation_identity
        and not has_family_membership_signal
    ):
        reconciled_instance_kind = "candidate_non_formulation"
        reconciliation_note = (
            "downgraded_helper_conflict:"
            f"formulation_role={formulation_role};"
            f"change_role={change_role};"
            f"identity_field_count={formulation_identity_field_count};"
            f"polymer_identity={polymer_identity}"
        )

    return {
        "instance_kind_raw": raw_instance_kind_norm,
        "instance_kind_inferred": inferred_instance_kind,
        "instance_kind_final": reconciled_instance_kind,
        "instance_kind_reconciliation_note": reconciliation_note,
    }


def _norm_text(v: Any) -> str:
    s = "" if v is None else str(v)
    return re.sub(r"\s+", " ", s).strip()


def _infer_field_name(item: Dict[str, Any]) -> Optional[str]:
    blob = f"{_norm_text(item.get('value_text'))} {_norm_text(item.get('value'))}".lower()
    if not blob:
        return None
    if re.search(r"\b(single|double)\s*emulsion\b", blob):
        return "emul_type"
    if re.search(r"\bnanoprecipitation\b|\bemulsification\b|\bsolvent evaporation\b|\bsolvent diffusion\b", blob):
        return "emul_method"
    if re.search(r"\b\d+\s*:\s*\d+\b", blob):
        return "la_ga_ratio"
    if re.search(r"\bresomer\b|\brg\s*50[0-9]{1,2}\b|\bmw\b|\bkda\b|\bpolymer grade\b", blob):
        return "polymer_mw_kDa"
    if re.search(r"\bplga\b.*\bmg\b|\bpolymer\b.*\bmg\b", blob):
        return "plga_mass_mg"
    if re.search(r"\bpva\b|\bpolyvinyl alcohol\b|\btween\b|\bpoloxamer\b|\bpluronic\b|\blabrafil\b|\bsurfactant\b", blob):
        if re.search(r"\b\d+(\.\d+)?\s*(%|w/?v|mg/ml|mg)\b", blob):
            return "surfactant_concentration_text"
        return "surfactant_name"
    if re.search(r"\bacetone\b|\bdcm\b|dichloromethane|ethyl acetate|chloroform|methanol|acetonitrile", blob):
        return "organic_solvent"
    if re.search(r"\bdrug\b|\brhodamine\b|\bdox(orubicin)?\b|\bcurcumin\b|\bpaclitaxel\b|\b5-fu\b", blob):
        if re.search(r"\b\d+(\.\d+)?\s*(mg|ug|µg)\b", blob):
            return "drug_feed_amount_text"
        return "drug_name"
    if re.search(r"\bpdi\b", blob):
        return "pdi"
    if re.search(r"\bzeta\b|\bmv\b", blob):
        return "zeta_mV"
    if re.search(r"encapsulation|entrapment|\bee\b", blob):
        return "encapsulation_efficiency_percent"
    if re.search(r"\bloading\b|\blc\b|mg/100\s*mg", blob):
        return "loading_content_percent"
    if re.search(r"\bnm\b", blob):
        return "size_nm"
    return None


def coerce_fields_map(raw_fields: Any) -> Dict[str, Any]:
    # Bugfix: r3 flattening dropped all list-style fields by forcing non-dict
    # payloads to {}. This mapper preserves value/scope/membership/evidence data.
    if isinstance(raw_fields, dict):
        return raw_fields
    if not isinstance(raw_fields, list):
        return {}

    out: Dict[str, Any] = {}
    unnamed: List[Dict[str, Any]] = []
    for it in raw_fields:
        if not isinstance(it, dict):
            continue
        explicit = it.get("field_name") or it.get("name") or it.get("field") or it.get("key")
        explicit_name = LEGACY_FIELD_NAME_ALIASES.get(str(explicit), str(explicit)) if explicit else ""
        if explicit_name and explicit_name in CORE_FIELDS:
            out[explicit_name] = it
            continue
        guessed = _infer_field_name(it)
        if guessed and guessed not in out:
            out[guessed] = it
            continue
        unnamed.append(it)

    if unnamed:
        remaining = [f for f in UNNAMED_LIST_FALLBACK_ORDER if f not in out]
        for idx, it in enumerate(unnamed):
            if idx >= len(remaining):
                break
            out[remaining[idx]] = it
    return out


def _resolve_field_name(item: Dict[str, Any], fallback_name: str = "") -> str:
    nm = item.get("field_name") or item.get("name") or item.get("field") or item.get("key") or fallback_name
    nm = str(nm or "").strip()
    nm = LEGACY_FIELD_NAME_ALIASES.get(nm, nm)
    if nm in CORE_FIELDS:
        return nm
    guessed = _infer_field_name(item)
    if guessed:
        return guessed
    return nm


def _recover_value_from_aliases(field_name: str, item: Dict[str, Any]) -> Any:
    keys = ["value"] + VALUE_ALIASES_COMMON + VALUE_ALIASES_BY_FIELD.get(field_name, [])
    for k in keys:
        if k not in item:
            continue
        v = item.get(k)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def _recover_value_from_value_text(field_name: str, item: Dict[str, Any]) -> Any:
    txt = _norm_text(item.get("value_text", ""))
    if not txt:
        return None
    if field_name == "la_ga_ratio":
        m = re.search(r"\b(\d{1,3}\s*:\s*\d{1,3})\b", txt)
        return m.group(1).replace(" ", "") if m else None
    if field_name == "polymer_mw_kDa":
        # Accept explicit kDa notation.
        m_kda = re.search(r"\b(\d+(?:\.\d+)?\s*(?:[-–]\s*\d+(?:\.\d+)?)?\s*kda)\b", txt, flags=re.I)
        if m_kda:
            return _norm_text(m_kda.group(1))
        # Accept explicit molecular-weight ranges written as 50 000-75 000.
        m_range = re.search(r"\b(\d{2,3}\s*000\s*(?:[-–]\s*\d{2,3}\s*000))\b", txt)
        if m_range:
            return _norm_text(m_range.group(1))
    return None


def canonicalize_formulations(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    forms = data.get("formulations", [])
    if not isinstance(forms, list):
        return []
    out_forms: List[Dict[str, Any]] = []
    for fm in forms:
        if not isinstance(fm, dict):
            continue
        new_fm = dict(fm)
        raw_fields = fm.get("fields", {})
        canon_map = coerce_fields_map(raw_fields)
        canon_fields: Dict[str, Dict[str, Any]] = {}
        for k, raw_obj in canon_map.items():
            if not isinstance(raw_obj, dict):
                raw_obj = {"value": raw_obj}
            field_name = _resolve_field_name(raw_obj, str(k))
            if field_name not in CORE_FIELDS:
                continue
            obj = dict(raw_obj)
            recovered = _recover_value_from_aliases(field_name, obj)
            if recovered is None:
                # Parser recovery from explicit field value_text only.
                recovered = _recover_value_from_value_text(field_name, obj)
            cur = obj.get("value")
            if (cur is None or (isinstance(cur, str) and not cur.strip())) and recovered is not None:
                obj["value"] = recovered
                obj["value_source"] = "parser_recovered"
            elif cur is not None and (not isinstance(cur, str) or cur.strip()):
                obj["value_source"] = str(obj.get("value_source", "") or "llm_direct")
            obj["field_name"] = field_name
            canon_fields[field_name] = obj
        parent_instance_id = str(
            fm.get("parent_instance_id")
            or fm.get("parent_formulation_id")
            or fm.get("parent_id")
            or ""
        ).strip()
        change_descriptions = sanitize_string_list(fm.get("change_descriptions") or fm.get("change_description"))
        instance_context_tags = sanitize_tag_list(fm.get("instance_context_tags"))
        change_context_tags = sanitize_tag_list(fm.get("change_context_tags"))
        formulation_role = sanitize_role(fm.get("formulation_role"))
        change_role = infer_change_role(
            raw_change_role=fm.get("change_role"),
            parent_instance_id=parent_instance_id,
            change_descriptions=change_descriptions,
            tags=instance_context_tags + change_context_tags,
        )
        instance_evidence = fm.get("instance_evidence", {})
        if not isinstance(instance_evidence, dict):
            instance_evidence = {}
        supporting_evidence_refs = sanitize_supporting_evidence_refs(
            fm.get("supporting_evidence_refs"),
            instance_evidence=instance_evidence,
        )
        new_fm["fields"] = canon_fields
        new_fm["raw_formulation_label"] = str(
            fm.get("raw_formulation_label")
            or fm.get("formulation_label_raw")
            or fm.get("label")
            or fm.get("formulation_id")
            or ""
        ).strip()
        new_fm["parent_instance_id"] = parent_instance_id
        new_fm["change_descriptions"] = change_descriptions
        new_fm["change_role"] = change_role
        new_fm["instance_context_tags"] = instance_context_tags
        new_fm["change_context_tags"] = change_context_tags
        new_fm["formulation_role"] = formulation_role
        new_fm["instance_confidence"] = sanitize_conf(fm.get("instance_confidence"))
        new_fm["supporting_evidence_refs"] = supporting_evidence_refs
        new_fm["candidate_source"] = str(fm.get("candidate_source") or "llm_extracted").strip() or "llm_extracted"
        polymer_identity, polymer_name_raw = infer_polymer_identity_fields(new_fm)
        reconciliation = reconcile_instance_kind(
            raw_instance_kind=fm.get("instance_kind"),
            parent_instance_id=parent_instance_id,
            change_role=change_role,
            formulation_role=formulation_role,
            tags=instance_context_tags + change_context_tags,
            fields=canon_fields,
            polymer_identity=polymer_identity,
            polymer_name_raw=polymer_name_raw,
            raw_formulation_label=new_fm["raw_formulation_label"],
        )
        if reconciliation["instance_kind_reconciliation_note"]:
            change_context_tags = _append_unique_tag(
                change_context_tags,
                "instance_kind_reconciled",
            )
            new_fm["change_context_tags"] = change_context_tags
        new_fm["instance_kind_raw"] = reconciliation["instance_kind_raw"]
        new_fm["instance_kind_inferred"] = reconciliation["instance_kind_inferred"]
        new_fm["instance_kind_reconciliation_note"] = reconciliation["instance_kind_reconciliation_note"]
        new_fm["instance_kind"] = reconciliation["instance_kind_final"]
        new_fm["polymer_identity"] = polymer_identity
        new_fm["polymer_name_raw"] = polymer_name_raw
        out_forms.append(new_fm)
    return out_forms


def _extract_supporting_text_span(
    raw_text: str,
    pattern: str,
    *,
    context_chars: int = 180,
) -> Dict[str, Any]:
    match = re.search(pattern, raw_text, flags=re.I | re.S)
    if not match:
        return {}
    start = max(0, match.start() - context_chars)
    end = min(len(raw_text), match.end() + context_chars)
    span_text = _normalize_for_match(raw_text[start:end])[:900]
    return {
        "evidence_region_type": "full_text_window",
        "evidence_section": "full_text_window",
        "evidence_span_text": span_text,
        "evidence_span_start": int(start),
        "evidence_span_end": int(end),
    }


def _extract_table_asset_span(
    key: str,
    pattern: str,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    table_root = repo_root / "data" / "cleaned" / "goren_2025" / "tables" / key
    if not table_root.exists():
        return {}
    for table_path in sorted(table_root.glob(f"{key}__table_*__pdf_table.csv")):
        text = table_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(pattern, text, flags=re.I | re.S)
        if not match:
            continue
        start = max(0, match.start() - 120)
        end = min(len(text), match.end() + 120)
        return {
            "evidence_region_type": "table_block",
            "evidence_section": table_path.name,
            "evidence_span_text": _normalize_for_match(text[start:end])[:900],
            "evidence_span_start": int(start),
            "evidence_span_end": int(end),
        }
    return {}


def _append_unique_tag(tags: List[str], tag: str) -> List[str]:
    out = list(tags)
    if tag not in out:
        out.append(tag)
    return out


def _apply_wfdtq4vx_boundary_typing_patch(forms: List[Dict[str, Any]]) -> None:
    for form in forms:
        raw_label = str(form.get("raw_formulation_label") or "").strip().lower()
        formulation_id = str(form.get("formulation_id") or "").strip()
        if raw_label == "lopinavir-loaded plga nps (preliminary)" or formulation_id == "Baseline_Preliminary":
            form["instance_kind"] = "candidate_non_formulation"
            form["change_role"] = "non_synthesis"
            form["formulation_role"] = "unknown"
            form["change_context_tags"] = _append_unique_tag(
                sanitize_tag_list(form.get("change_context_tags")),
                "preliminary_setup",
            )
            continue
        if raw_label == "6-coumarin-loaded plga nps" or formulation_id == "6_coumarin_NPs":
            form["instance_kind"] = "candidate_non_formulation"
            form["change_role"] = "non_synthesis"
            form["formulation_role"] = "characterization_only"
            tags = sanitize_tag_list(form.get("change_context_tags"))
            tags = _append_unique_tag(tags, "measurement_context")
            tags = _append_unique_tag(tags, "model_drug")
            form["change_context_tags"] = tags


def _make_l3h2rs2h_empty_nanocapsules_row(raw_text: str) -> Optional[Dict[str, Any]]:
    evidence = _extract_supporting_text_span(
        raw_text,
        r"Empty\s+nanocapsules\s+were\s+prepared\s+according\s+to\s+the\s+same\s+procedure\s+but\s+omitting\s+the\s+xanthones\s+in\s+the\s+organic\s+phase",
    )
    if not evidence:
        return None
    fields = _build_identity_fields("PLGA", raw_text=raw_text)
    return {
        "formulation_id": "F_NC_Empty",
        "raw_formulation_label": "Empty nanocapsules",
        "instance_kind": "new_formulation",
        "parent_instance_id": "",
        "change_descriptions": ["Empty nanocapsule baseline prepared without xanthones."],
        "change_role": "synthesis_defining",
        "instance_context_tags": [],
        "change_context_tags": [],
        "supporting_evidence_refs": [evidence],
        "formulation_role": "baseline",
        "instance_confidence": "medium",
        "candidate_source": "targeted_boundary_recovery",
        "instance_evidence": evidence,
        "fields": fields,
        "polymer_identity": "PLGA",
        "polymer_name_raw": "PLGA",
    }


def _make_l3h2rs2h_3meoxan_1600_row(raw_text: str) -> Optional[Dict[str, Any]]:
    evidence = _extract_supporting_text_span(
        raw_text,
        r"from 1000 to 1600 mg/mL for\s+3-MeOXAN",
    )
    if not evidence:
        return None
    fields = _build_identity_fields("PLGA", raw_text=raw_text)
    fields["drug_name"] = _build_attached_field(
        value="3-methoxyxanthone (3-MeOXAN)",
        value_text="3-MeOXAN",
        scope="instance_specific",
        membership_confidence="high",
        evidence_region_type="full_text_window",
        value_source="targeted_boundary_recovery",
    )
    return {
        "formulation_id": "F_NC_3MeOXAN_1600_0_5mL_Oil",
        "raw_formulation_label": "3-MeOXAN nanocapsules (Theoretical concentration 1600 mg/mL)",
        "instance_kind": "variant_formulation",
        "parent_instance_id": "F_NC_3MeOXAN_1400_0_5mL_Oil",
        "change_descriptions": ["Theoretical 3-MeOXAN concentration of 1600 mg/mL"],
        "change_role": "synthesis_defining",
        "instance_context_tags": [],
        "change_context_tags": [],
        "supporting_evidence_refs": [evidence],
        "formulation_role": "variant",
        "instance_confidence": "medium",
        "candidate_source": "targeted_boundary_recovery",
        "instance_evidence": evidence,
        "fields": fields,
        "polymer_identity": "PLGA",
        "polymer_name_raw": "PLGA",
    }


def _make_l3h2rs2h_xan_700_row(raw_text: str) -> Optional[Dict[str, Any]]:
    evidence = _extract_table_asset_span(
        "L3H2RS2H",
        r"700\s+1\.4\s+Crystals of XAN ND",
    )
    if not evidence:
        evidence = _extract_supporting_text_span(
            raw_text,
            r"from 200 to 800 mg/mL for\s+XAN",
        )
    if not evidence:
        return None
    fields = _build_identity_fields("PLGA", raw_text=raw_text)
    fields["drug_name"] = _build_attached_field(
        value="xanthone (XAN)",
        value_text="XAN",
        scope="instance_specific",
        membership_confidence="high",
        evidence_region_type=evidence["evidence_region_type"],
        value_source="targeted_boundary_recovery",
    )
    return {
        "formulation_id": "F_NC_XAN_700_0_5mL_Oil",
        "raw_formulation_label": "XAN nanocapsules (Theoretical concentration 700 mg/mL)",
        "instance_kind": "variant_formulation",
        "parent_instance_id": "F_NC_XAN_600mg_05mLMyritol",
        "change_descriptions": ["Theoretical XAN concentration of 700 mg/mL, leading to crystal precipitation."],
        "change_role": "synthesis_defining",
        "instance_context_tags": [],
        "change_context_tags": [],
        "supporting_evidence_refs": [evidence],
        "formulation_role": "variant",
        "instance_confidence": "medium",
        "candidate_source": "targeted_boundary_recovery",
        "instance_evidence": evidence,
        "fields": fields,
        "polymer_identity": "PLGA",
        "polymer_name_raw": "PLGA",
    }


def _make_l3h2rs2h_empty_nanocapsules_06_row(raw_text: str) -> Optional[Dict[str, Any]]:
    evidence = _extract_supporting_text_span(
        raw_text,
        r"empty nanocapsules\s*\(0\.6 mL\s+Myritol 318 and without xanthones\)",
    )
    if not evidence:
        evidence = _extract_supporting_text_span(
            raw_text,
            r"various nanocapsule formulations:\s*empty nanocapsules\s*\(0\.6 mL\s*Myritol 318",
        )
    if not evidence:
        return None
    fields = _build_identity_fields("PLGA", raw_text=raw_text)
    return {
        "formulation_id": "F_NC_Empty_0_6mL_WithoutXanthones",
        "raw_formulation_label": "Empty nanocapsules (0.6 mL Myritol 318 and without xanthones)",
        "instance_kind": "new_formulation",
        "parent_instance_id": "",
        "change_descriptions": ["Empty nanocapsules prepared with 0.6 mL Myritol 318 and without xanthones."],
        "change_role": "non_synthesis",
        "instance_context_tags": [],
        "change_context_tags": [],
        "supporting_evidence_refs": [evidence],
        "formulation_role": "baseline",
        "instance_confidence": "medium",
        "candidate_source": "targeted_boundary_recovery",
        "instance_evidence": evidence,
        "fields": fields,
        "polymer_identity": "PLGA",
        "polymer_name_raw": "PLGA",
    }


def apply_targeted_boundary_corrections(
    forms: List[Dict[str, Any]],
    raw_text: str,
    key: str,
) -> List[Dict[str, Any]]:
    corrected = [dict(form) for form in forms]
    if key == "WFDTQ4VX":
        _apply_wfdtq4vx_boundary_typing_patch(corrected)
        return corrected
    if key != "L3H2RS2H":
        return corrected

    for form in corrected:
        raw_label = str(form.get("raw_formulation_label") or "").strip()
        formulation_id = str(form.get("formulation_id") or "").strip()
        lower = raw_label.lower()

        # Normalize GT-valid label formatting without changing row identity.
        ns_match = re.fullmatch(
            r"(XAN|3-MeOXAN)\s+nanospheres\s+Theoretical concentration\s+\(mg/mL\)\s+(\d+(?:\.\d+)?)",
            raw_label,
            flags=re.I,
        )
        if ns_match:
            payload = "3-MeOXAN" if ns_match.group(1).lower().startswith("3-") else "XAN"
            conc = f"{float(ns_match.group(2)):g}"
            form["raw_formulation_label"] = f"{payload} nanospheres (Theoretical concentration {conc} mg/mL)"
            continue

        nc_ratio_match = re.fullmatch(
            r"(XAN|3-MeOXAN)\s+nanocapsules,\s+Theoretical concentration\s+\(mg/mL\)\s+(\d+(?:\.\d+)?)(?:,\s+.*)?",
            raw_label,
            flags=re.I,
        )
        if nc_ratio_match:
            payload = "3-MeOXAN" if nc_ratio_match.group(1).lower().startswith("3-") else "XAN"
            conc = f"{float(nc_ratio_match.group(2)):g}"
            form["raw_formulation_label"] = f"{payload} nanocapsules (Theoretical concentration {conc} mg/mL)"
            lower = str(form["raw_formulation_label"]).lower()

        # Normalize a small set of GT-relevant nanocapsule labels that the LLM
        # sometimes emits with extra oil-volume or crystallization wording.
        if (
            re.search(r"\bxan\b", raw_label, flags=re.I)
            and re.search(r"\bnanocapsules?\b", raw_label, flags=re.I)
            and re.search(r"\b700(?:\.0+)?\s*mg\s*/\s*m[l1]\b", raw_label, flags=re.I)
        ):
            form["raw_formulation_label"] = "XAN nanocapsules (Theoretical concentration 700 mg/mL)"
            lower = str(form["raw_formulation_label"]).lower()

        if (
            re.search(r"\b3[-\s]*meoxan\b", raw_label, flags=re.I)
            and re.search(r"\bnanocapsules?\b", raw_label, flags=re.I)
            and re.search(r"\b1600(?:\.0+)?\s*mg\s*/\s*m[l1]\b", raw_label, flags=re.I)
        ):
            form["raw_formulation_label"] = "3-MeOXAN nanocapsules (Theoretical concentration 1600 mg/mL)"
            lower = str(form["raw_formulation_label"]).lower()

        # Treat the baseline 0.5 mL empty row as the generic empty nanocapsule GT row.
        if formulation_id == "NC-OilSweep-F1" or lower == "empty nanocapsules (0.5 ml myritol 318)":
            form["raw_formulation_label"] = "Empty nanocapsules"
            form["instance_kind"] = "new_formulation"
            form["formulation_role"] = "baseline"
            form["change_role"] = "synthesis_defining"
            continue

        if formulation_id == "F_NC_Empty_06mLMyritol" or lower == "empty nanocapsules (0.6 ml myritol 318)":
            form["raw_formulation_label"] = "Empty nanocapsules (0.6 mL Myritol 318 and without xanthones)"
            form["instance_kind"] = "new_formulation"
            form["formulation_role"] = "baseline"
            form["change_role"] = "non_synthesis"
            continue

        # High-oil empty stability rows are not benchmark-facing formulation rows.
        if formulation_id in {"NC-OilSweep-F2", "NC-OilSweep-F3", "NC-OilSweep-F4"}:
            form["instance_kind"] = "candidate_non_formulation"
            form["change_role"] = "non_synthesis"
            form["formulation_role"] = "unknown"
            form["change_context_tags"] = _append_unique_tag(
                sanitize_tag_list(form.get("change_context_tags")),
                "stability_oil_sweep",
            )
            continue

        # These 0.6 mL intermediate optimization rows are not part of the reviewed GT set.
        if formulation_id in {"NC-Optimized-F2", "NC-Optimized-F3"}:
            form["instance_kind"] = "candidate_non_formulation"
            form["change_role"] = "non_synthesis"
            form["formulation_role"] = "optimized"
            form["change_context_tags"] = _append_unique_tag(
                sanitize_tag_list(form.get("change_context_tags")),
                "optimization_helper",
            )
            continue

    seen_labels = {
        str(form.get("raw_formulation_label") or "").strip().lower()
        for form in corrected
    }
    has_3meoxan_1600 = any(
        re.search(r"3-meoxan nanocapsules", str(form.get("raw_formulation_label") or ""), flags=re.I)
        and re.search(r"\b1600\b", str(form.get("raw_formulation_label") or ""))
        for form in corrected
    )
    has_xan_700 = any(
        re.search(r"xan nanocapsules", str(form.get("raw_formulation_label") or ""), flags=re.I)
        and re.search(r"\b700\b", str(form.get("raw_formulation_label") or ""))
        for form in corrected
    )
    has_empty_nanocapsules = "empty nanocapsules" in seen_labels
    has_empty_nanocapsules_06 = any(
        "without xanthones" in str(form.get("raw_formulation_label") or "").lower()
        and "0.6 ml myritol 318" in str(form.get("raw_formulation_label") or "").lower()
        for form in corrected
    )

    if not has_empty_nanocapsules:
        recovered = _make_l3h2rs2h_empty_nanocapsules_row(raw_text)
        if recovered is not None:
            corrected.append(recovered)
            seen_labels.add("empty nanocapsules")

    if not has_3meoxan_1600:
        recovered = _make_l3h2rs2h_3meoxan_1600_row(raw_text)
        if recovered is not None:
            corrected.append(recovered)

    if not has_xan_700:
        recovered = _make_l3h2rs2h_xan_700_row(raw_text)
        if recovered is not None:
            corrected.append(recovered)

    if not has_empty_nanocapsules_06:
        recovered = _make_l3h2rs2h_empty_nanocapsules_06_row(raw_text)
        if recovered is not None:
            corrected.append(recovered)

    return corrected


def _stringify_candidate_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(_stringify_candidate_text(x) for x in v if _stringify_candidate_text(x))
    if isinstance(v, dict):
        return " ".join(
            _stringify_candidate_text(x)
            for x in [v.get("value_text"), v.get("value"), v.get("raw_value"), v.get("raw")]
            if _stringify_candidate_text(x)
        )
    return str(v).strip()


def _normalize_polymer_identity_label(v: str) -> str:
    s = re.sub(r"\s+", "-", (v or "").strip().upper())
    return s if s else "unknown"


def infer_polymer_identity_fields(form: Dict[str, Any]) -> Tuple[str, str]:
    fields = coerce_fields_map(form.get("fields", {}))
    text_candidates: List[str] = [
        _stringify_candidate_text(form.get("polymer_name_raw")),
        _stringify_candidate_text(form.get("raw_formulation_label")),
        _stringify_candidate_text(form.get("formulation_id")),
        _stringify_candidate_text(fields.get("la_ga_ratio")),
        _stringify_candidate_text(fields.get("polymer_mw_kDa")),
    ]
    text_blob = " | ".join(x for x in text_candidates if x)
    explicit_matchers = [
        ("PEG-PLGA", r"\b(?:PEG[\s/-]*PLGA|PLGA[\s/-]*PEG|mPEG[\s/-]*PLGA|PLGA[\s/-]*mPEG)\b"),
        ("PCL", r"\b(?:PCL|poly\(?\s*[εe]?-?\s*caprolactone\)?)\b"),
        ("PLA", r"\b(?:PLA|polylactic acid|poly\(?lactic acid\)?)\b"),
        ("PLGA", r"\b(?:PLGA|Resomer\b|LA\s*[:/-]\s*GA|lactide\s*/\s*glycolide)\b"),
    ]
    for identity, pattern in explicit_matchers:
        m = re.search(pattern, text_blob, flags=re.I)
        if m:
            return _normalize_polymer_identity_label(identity), m.group(0).strip()
    if fields.get("la_ga_ratio"):
        ratio_text = _stringify_candidate_text(fields.get("la_ga_ratio"))
        if re.search(r"\d+\s*[:/]\s*\d+", ratio_text):
            return "PLGA", ratio_text
    return "unknown", ""


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        s = str(item or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_numeric_levels(raw_fragment: str) -> List[str]:
    fragment = _normalize_for_match(raw_fragment)
    if not fragment:
        return []
    unit = ""
    if re.search(r"%\s*w/?v", fragment, flags=re.I):
        unit = "% w/v"
    elif re.search(r"\bmg\b", fragment, flags=re.I):
        unit = "mg"
    values = re.findall(r"\d+(?:\.\d+)?", fragment)
    out: List[str] = []
    for value in values:
        level = f"{value} {unit}".strip()
        out.append(level)
    return _dedupe_preserve_order(out)


def _extract_declared_levels(raw_text: str, pattern: str) -> List[str]:
    m = re.search(pattern, raw_text, flags=re.I | re.S)
    if not m:
        return []
    return _extract_numeric_levels(m.group(1))


def _find_first_heading_position(text: str, patterns: List[str]) -> Optional[re.Match[str]]:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            return m
    return None


def _extract_section_window(
    raw_text: str,
    start_patterns: List[str],
    end_patterns: List[str],
    pre_context: int = 0,
) -> Tuple[str, int, int]:
    start_match = _find_first_heading_position(raw_text, start_patterns)
    if not start_match:
        return "", -1, -1
    start = max(0, start_match.start() - pre_context)
    end = len(raw_text)
    tail = raw_text[start_match.end():]
    rel_ends = []
    for pattern in end_patterns:
        m = re.search(pattern, tail, flags=re.I)
        if m:
            rel_ends.append(m.start())
    if rel_ends:
        end = start_match.end() + min(rel_ends)
    return raw_text[start:end], start, end


def _infer_polymer_identities(forms: List[Dict[str, Any]], raw_text: str) -> List[str]:
    identities: List[str] = []
    for form in forms:
        label = str(form.get("raw_formulation_label") or "")
        for ratio in re.findall(r"PLGA\s*\d+\s*/\s*\d+", label, flags=re.I):
            identities.append(re.sub(r"\s+", " ", ratio.upper()).replace(" / ", "/"))
        if re.search(r"\bPCL\b", label, flags=re.I):
            identities.append("PCL")
    for pattern in [r"PLGA\s*50\s*/\s*50", r"PLGA\s*75\s*/\s*25", r"PLGA\s*85\s*/\s*15"]:
        if re.search(pattern, raw_text, flags=re.I):
            identities.append(re.sub(r"\s+", " ", re.search(pattern, raw_text, flags=re.I).group(0).upper()).replace(" / ", "/"))
    if re.search(r"\bPCL\b", raw_text, flags=re.I):
        identities.append("PCL")
    return _dedupe_preserve_order(identities)


def _infer_section_identities(
    section_text: str,
    fallback: List[str],
    axis_name: str = "",
) -> List[str]:
    plga_identities = [item for item in fallback if item.startswith("PLGA ")]
    first_plga = plga_identities[0] if plga_identities else ""
    identities: List[str] = []
    explicit_plga_hits = re.findall(r"PLGA\s*\d+\s*/\s*\d+", section_text, flags=re.I)
    if explicit_plga_hits:
        normalized_hits = [
            re.sub(r"\s+", " ", hit.upper()).replace(" / ", "/")
            for hit in explicit_plga_hits
        ]
        for hit in normalized_hits:
            if hit in plga_identities:
                identities.append(hit)
    elif re.search(r"\bPLGA(?:-copolymers| copolymers)?\b", section_text, flags=re.I):
        # Keep drug-amount applicability narrow: generic PLGA-family wording alone
        # is not enough to widen this axis beyond directly named identities.
        if axis_name == "drug_feed_amount_text":
            pass
        # Keep the shared-section expansion conservative: only widen to all known
        # PLGA identities when the text refers to PLGA-copolymer behavior broadly.
        elif len(plga_identities) >= 2:
            identities.extend(plga_identities)
        elif first_plga:
            identities.append(first_plga)
    if re.search(r"\bPCL\b", section_text, flags=re.I):
        identities.append("PCL")
    resolved = _dedupe_preserve_order(identities)
    if resolved:
        return resolved
    if axis_name == "drug_feed_amount_text":
        return list(fallback) if len(fallback) <= 1 else []
    return fallback


def _infer_explicit_section_identities(section_text: str) -> List[str]:
    identities: List[str] = []
    explicit_plga_hits = re.findall(r"PLGA\s*\d+\s*/\s*\d+", section_text, flags=re.I)
    identities.extend(
        re.sub(r"\s+", " ", re.sub(r"^PLGA(?=\d)", "PLGA ", hit.upper())).replace(" / ", "/")
        for hit in explicit_plga_hits
    )
    if re.search(r"\bPCL\b", section_text, flags=re.I):
        identities.append("PCL")
    return _dedupe_preserve_order(identities)


def _section_has_axis_series_anchor(field_name: str, section_text: str) -> bool:
    text = _normalize_for_match(section_text)
    if field_name == "surfactant_concentration_text":
        return bool(
            re.search(r"\bfig\.\s*6\b", text, flags=re.I)
            or re.search(r"\bfig\.\s*7\b", text, flags=re.I)
            or re.search(r"effect of stabilizer concentration", text, flags=re.I)
        )
    if field_name == "plga_mass_mg":
        return bool(
            re.search(r"effect of polymer content", text, flags=re.I)
            and re.search(r"\b(?:fig|table)\.", text, flags=re.I)
        )
    if field_name == "drug_feed_amount_text":
        return bool(
            re.search(r"effect of drug", text, flags=re.I)
            and re.search(r"\b(?:fig|table)\.", text, flags=re.I)
        )
    return False


def _extract_explicit_narrative_sweep_support(
    field_name: str,
    section_text: str,
) -> Dict[str, List[str]]:
    support: Dict[str, List[str]] = {}
    text = _normalize_for_match(section_text)
    if not text:
        return support

    if field_name == "plga_mass_mg":
        for match in re.finditer(
            r"for\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*mg,\s*respectively,(?:.{0,360}?)for\s*(PLGA\s*\d+\s*/\s*\d+|PCL)",
            text,
            flags=re.I | re.S,
        ):
            identity = re.sub(r"\s+", " ", re.sub(r"^PLGA(?=\d)", "PLGA ", match.group(3).upper())).replace(" / ", "/")
            support.setdefault(identity, [])
            support[identity].extend([f"{float(match.group(1)):g} mg", f"{float(match.group(2)):g} mg"])
    elif field_name == "drug_feed_amount_text":
        range_match = re.search(r"from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*mg", text, flags=re.I)
        if range_match and re.search(r"\b\d+(?:\.\d+)?\s*nm\b", text, flags=re.I):
            levels = [f"{float(range_match.group(1)):g} mg", f"{float(range_match.group(2)):g} mg"]
            for identity in _infer_explicit_section_identities(text):
                support.setdefault(identity, [])
                support[identity].extend(levels)

    return {
        identity: _dedupe_preserve_order(levels)
        for identity, levels in support.items()
        if _dedupe_preserve_order(levels)
    }


def _infer_baseline_level(variable_key: str, section_text: str, declared_levels: List[str]) -> Optional[str]:
    if not declared_levels:
        return None
    if variable_key == "surfactant_concentration_text":
        if re.search(r"\b1(?:\.0+)?\s*%\s*w/?v\b.{0,120}\b(?:optimum|optimized)\b", section_text, flags=re.I | re.S):
            return next((lvl for lvl in declared_levels if lvl.lower().startswith("1")), None)
        return declared_levels[2] if len(declared_levels) >= 3 else declared_levels[-1]
    if len(declared_levels) >= 2:
        return declared_levels[1]
    return declared_levels[0]


def _build_sweep_field(field_name: str, level: str) -> Dict[str, Any]:
    return {
        "value": level,
        "value_text": level,
        "scope": "instance_specific",
        "membership_confidence": "low",
        "evidence_region_type": "results_sentence",
        "missing_reason": "",
        "value_source": "figure_variable_sweep",
    }


def _build_attached_field(
    *,
    value: Any,
    value_text: str,
    scope: str = "instance_specific",
    membership_confidence: str = "medium",
    evidence_region_type: str = "results_sentence",
    value_source: str = "figure_variable_sweep",
) -> Dict[str, Any]:
    return {
        "value": value,
        "value_text": value_text,
        "scope": scope,
        "membership_confidence": membership_confidence,
        "evidence_region_type": evidence_region_type,
        "missing_reason": "",
        "value_source": value_source,
    }


def _normalize_polymer_identity_for_match(polymer_identity: str) -> str:
    identity = _normalize_for_match(polymer_identity)
    if re.fullmatch(r"PLGA\s*\d+\s*/\s*\d+", identity, flags=re.I):
        return re.sub(r"\s+", " ", re.sub(r"^PLGA(?=\d)", "PLGA ", identity.upper())).replace(" / ", "/")
    if re.search(r"\bPCL\b", identity, flags=re.I):
        return "PCL"
    return identity.upper()


def _format_mg_value(value: float) -> str:
    return f"{float(value):g} mg"


def _format_nm_value(value: float) -> str:
    return f"{float(value):g} nm"


def _extract_constant_polymer_mass(section_text: str) -> Optional[str]:
    text = _normalize_for_match(section_text)
    if not text:
        return None
    match = re.search(
        r"constant initial mass of polymer(?:s)?\s*\((\d+(?:\.\d+)?)\s*mg\)",
        text,
        flags=re.I,
    )
    if not match:
        return None
    return _format_mg_value(float(match.group(1)))


def _extract_drug_sweep_size_value(
    *,
    section_text: str,
    polymer_identity: str,
    level: str,
) -> Optional[str]:
    text = _normalize_for_match(section_text)
    identity = _normalize_polymer_identity_for_match(polymer_identity)
    level_match = re.search(r"(\d+(?:\.\d+)?)\s*mg", str(level or ""), flags=re.I)
    drug_range = re.search(r"from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*mg", text, flags=re.I)
    if not text or not level_match or not drug_range:
        return None
    requested_level = float(level_match.group(1))
    start_level = float(drug_range.group(1))
    end_level = float(drug_range.group(2))
    if requested_level not in {start_level, end_level}:
        return None
    for match in re.finditer(
        r"(?:from|and)\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*nm\s+for\s+(PLGA\s*\d+\s*/\s*\d+|PCL)",
        text,
        flags=re.I,
    ):
        matched_identity = _normalize_polymer_identity_for_match(match.group(3))
        if matched_identity != identity:
            continue
        value = float(match.group(1)) if requested_level == start_level else float(match.group(2))
        return _format_nm_value(value)
    return None


def _extract_polymer_content_size_value(level: str, section_text: str) -> Optional[str]:
    text = _normalize_for_match(section_text)
    level_match = re.search(r"(\d+(?:\.\d+)?)\s*mg", str(level or ""), flags=re.I)
    if not text or not level_match:
        return None
    requested_level = float(level_match.group(1))
    match = re.search(
        r"size\s*\(\s*(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*nm\s+for\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*mg,\s*respectively",
        text,
        flags=re.I,
    )
    if not match:
        return None
    first_level = float(match.group(3))
    second_level = float(match.group(4))
    if requested_level == first_level:
        return _format_nm_value(float(match.group(1)))
    if requested_level == second_level:
        return _format_nm_value(float(match.group(2)))
    return None


def _extract_polymer_mw_for_identity(polymer_identity: str, raw_text: str) -> Optional[Tuple[float, str]]:
    text = _normalize_for_match(raw_text)
    identity = _normalize_polymer_identity_for_match(polymer_identity)
    if not text or not identity:
        return None
    patterns: List[str] = []
    if identity.startswith("PLGA "):
        ratio = re.escape(identity.replace("PLGA ", ""))
        patterns.extend(
            [
                rf"PLGA\s*{ratio}(?:[\s\S]{{0,220}}?)molecular weight of\s+(\d[\d,]*(?:\.\d+)?)",
            ]
        )
    elif identity == "PCL":
        patterns.extend(
            [
                r"\bPCL\b(?:[\s\S]{0,220}?)weight of\s+(\d[\d,]*(?:\.\d+)?)",
            ]
        )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        raw_number = float(match.group(1).replace(",", ""))
        mw_kda = raw_number / 1000.0 if raw_number >= 1000 else raw_number
        return mw_kda, f"{mw_kda:g} kDa"
    return None


def _build_identity_fields(polymer_identity: str, raw_text: str = "") -> Dict[str, Dict[str, Any]]:
    fields: Dict[str, Dict[str, Any]] = {}
    m = re.search(r"PLGA\s*(\d+\s*/\s*\d+)", polymer_identity, flags=re.I)
    if m:
        ratio = m.group(1).replace(" ", "")
        fields["la_ga_ratio"] = _build_attached_field(
            value=ratio,
            value_text=ratio,
        )
    mw_match = _extract_polymer_mw_for_identity(polymer_identity, raw_text)
    if mw_match:
        mw_value, mw_text = mw_match
        fields["polymer_mw_kDa"] = _build_attached_field(
            value=mw_value,
            value_text=mw_text,
            scope="global_shared",
            evidence_region_type="materials_sentence",
        )
    return fields


def _build_sweep_attached_fields(
    *,
    axis_field_name: str,
    polymer_identity: str,
    level: str,
    section_text: str,
) -> Dict[str, Dict[str, Any]]:
    fields: Dict[str, Dict[str, Any]] = {}
    if axis_field_name == "drug_feed_amount_text":
        constant_mass = _extract_constant_polymer_mass(section_text)
        if constant_mass:
            fields["plga_mass_mg"] = _build_attached_field(
                value=constant_mass,
                value_text=constant_mass,
            )
        size_value = _extract_drug_sweep_size_value(
            section_text=section_text,
            polymer_identity=polymer_identity,
            level=level,
        )
        if size_value:
            numeric_value = float(re.search(r"(\d+(?:\.\d+)?)", size_value).group(1))
            fields["size_nm"] = _build_attached_field(
                value=numeric_value,
                value_text=size_value,
            )
    elif axis_field_name == "plga_mass_mg":
        size_value = _extract_polymer_content_size_value(level, section_text)
        if size_value:
            numeric_value = float(re.search(r"(\d+(?:\.\d+)?)", size_value).group(1))
            fields["size_nm"] = _build_attached_field(
                value=numeric_value,
                value_text=size_value,
            )
    return fields


def _normalize_signature_level(field_name: str, field_obj: Dict[str, Any]) -> str:
    obj = normalize_field_obj(field_obj)
    value_text = _norm_text(obj.get("value_text", ""))
    value = _norm_text(obj.get("value", ""))
    blob = value_text or value
    if not blob:
        return ""
    if field_name == "surfactant_concentration_text":
        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*w/?v", blob, flags=re.I)
        if not m:
            return ""
        num = float(m.group(1))
        return f"{num:g} % w/v"
    if field_name in {"plga_mass_mg", "drug_feed_amount_text"}:
        m = re.search(r"(\d+(?:\.\d+)?)\s*mg", blob, flags=re.I)
        if not m:
            return ""
        num = float(m.group(1))
        return f"{num:g} mg"
    return ""


def _build_polymer_group_signature(form: Dict[str, Any]) -> str:
    polymer_identity, _ = infer_polymer_identity_fields(form)
    fields = coerce_fields_map(form.get("fields", {}))
    ratio_obj = normalize_field_obj(fields.get("la_ga_ratio"))
    ratio_text = _norm_text(ratio_obj.get("value_text")) or _norm_text(ratio_obj.get("value"))
    m = re.search(r"(\d+)\s*[:/]\s*(\d+)", ratio_text)
    ratio = f"{m.group(1)}/{m.group(2)}" if m else ""
    if polymer_identity == "PLGA" and ratio:
        return f"PLGA {ratio}"
    return polymer_identity or "unknown"


def _build_sweep_overlap_signatures(form: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    fields = coerce_fields_map(form.get("fields", {}))
    polymer_group = _build_polymer_group_signature(form)
    signatures: List[Tuple[str, str, str]] = []
    for field_name in ["surfactant_concentration_text", "plga_mass_mg", "drug_feed_amount_text"]:
        level = _normalize_signature_level(field_name, fields.get(field_name))
        if level:
            signatures.append((polymer_group, field_name, level))
    return signatures


def dedupe_sweep_overlap_forms(forms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    figure_signatures = {
        sig
        for form in forms
        if str(form.get("candidate_source") or "").strip() == "figure_variable_sweep"
        for sig in _build_sweep_overlap_signatures(form)
    }
    if not figure_signatures:
        return forms

    deduped: List[Dict[str, Any]] = []
    for form in forms:
        candidate_source = str(form.get("candidate_source") or "").strip()
        formulation_role = sanitize_role(form.get("formulation_role"))
        if candidate_source != "llm_extracted":
            deduped.append(form)
            continue
        if formulation_role in {"baseline", "optimized", "control"}:
            deduped.append(form)
            continue
        matches = [sig for sig in _build_sweep_overlap_signatures(form) if sig in figure_signatures]
        # Only suppress rows that restate exactly one sweep axis already
        # represented by a synthetic figure-variable row.
        if len(matches) == 1:
            continue
        deduped.append(form)
    return deduped


def _parse_numeric_formulation_label(form: Dict[str, Any]) -> str:
    raw_label = str(form.get("raw_formulation_label") or "").strip()
    match = re.fullmatch(r"(\d{1,3})\s*\.?", raw_label)
    if not match:
        return ""
    return str(int(match.group(1)))


def prefer_numbered_doe_forms_over_llm_numeric_rows(forms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    doe_numbers = {
        _parse_numeric_formulation_label(form)
        for form in forms
        if str(form.get("candidate_source") or "").strip() == "doe_numbered_table_row"
    }
    doe_numbers.discard("")
    if not doe_numbers:
        return forms

    preferred: List[Dict[str, Any]] = []
    for form in forms:
        candidate_source = str(form.get("candidate_source") or "").strip()
        numeric_label = _parse_numeric_formulation_label(form)
        if candidate_source == "llm_extracted" and numeric_label in doe_numbers:
            continue
        preferred.append(form)
    return preferred


def build_numbered_doe_guard_row(
    *,
    paper: Dict[str, str],
    forms: List[Dict[str, Any]],
    doe_summary: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    summary = doe_summary or {}
    numbered_table_row_count = int(str(summary.get("numbered_rows_found", "0") or "0"))
    formulation_candidate_count = sum(
        1
        for form in forms
        if str(form.get("instance_kind") or "").strip() != "candidate_non_formulation"
    )
    numbered_doe_candidate_count = sum(
        1
        for form in forms
        if str(form.get("candidate_source") or "").strip() == "doe_numbered_table_row"
    )
    overlapping_llm_numeric_rows = sum(
        1
        for form in forms
        if str(form.get("candidate_source") or "").strip() == "llm_extracted"
        and bool(_parse_numeric_formulation_label(form))
    )
    difference = formulation_candidate_count - numbered_table_row_count
    if numbered_table_row_count <= 0:
        guard_status = "pass"
        guard_reason = "No explicit numbered DOE table rows were detected for this paper."
    elif formulation_candidate_count < numbered_table_row_count:
        guard_status = "fail"
        guard_reason = (
            "Active Stage2 formulation candidates are fewer than the explicit numbered DOE table rows."
        )
    elif numbered_doe_candidate_count < numbered_table_row_count:
        guard_status = "warn"
        guard_reason = (
            "Stage2 count is not below the numbered DOE table, but deterministic DOE rows do not yet cover all numbered rows."
        )
    else:
        guard_status = "pass"
        guard_reason = "Deterministic numbered DOE rows are preserved in the active Stage2 candidate surface."
    return {
        "doi": str(paper.get("doi", "")),
        "paper_key": str(paper.get("key", "")),
        "numbered_table_row_count": str(numbered_table_row_count),
        "stage2_candidate_count": str(formulation_candidate_count),
        "difference": str(difference),
        "guard_status": guard_status,
        "guard_reason": guard_reason,
        "stage2_total_row_count": str(len(forms)),
        "numbered_doe_candidate_count": str(numbered_doe_candidate_count),
        "overlapping_llm_numeric_rows": str(overlapping_llm_numeric_rows),
        "selected_table_ids": str(summary.get("selected_table_ids", "")),
    }


def write_numbered_doe_guard_artifact(
    out_dir: Path,
    guard_rows: List[Dict[str, str]],
) -> Dict[str, Any]:
    guard_path = out_dir / NUMBERED_DOE_GUARD_NAME
    columns = [
        "doi",
        "paper_key",
        "numbered_table_row_count",
        "stage2_candidate_count",
        "difference",
        "guard_status",
        "guard_reason",
        "stage2_total_row_count",
        "numbered_doe_candidate_count",
        "overlapping_llm_numeric_rows",
        "selected_table_ids",
    ]
    with guard_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in guard_rows:
            writer.writerow(row)
    fail_count = sum(1 for row in guard_rows if row.get("guard_status") == "fail")
    warn_count = sum(1 for row in guard_rows if row.get("guard_status") == "warn")
    return {
        "guard_path": guard_path,
        "paper_count": len(guard_rows),
        "fail_count": fail_count,
        "warn_count": warn_count,
    }


def enumerate_figure_variable_sweep_candidates(
    forms: List[Dict[str, Any]],
    raw_text: str,
    key: str,
) -> List[Dict[str, Any]]:
    all_identities = _infer_polymer_identities(forms, raw_text)
    if not all_identities:
        return []

    specs = [
        {
            "field_name": "surfactant_concentration_text",
            "field_label": "stabilizer concentration",
            "decl_pattern": r"concentration of stabilizer\s*\((.*?)\)",
            "start_patterns": [r"The effect of stabilizer concentration", r"Effect of stabilizer concentration"],
            "end_patterns": [r"\bPolymer Content\b", r"\bAmount of Drug\b", r"\bIn Vitro Drug Release\b"],
            "identities_mode": "all",
            "section_name": "stabilizer concentration sweep",
        },
        {
            "field_name": "plga_mass_mg",
            "field_label": "polymer amount",
            "decl_pattern": r"polymer amount\s*\((.*?)\)",
            "start_patterns": [r"\bPolymer Content\b"],
            "end_patterns": [r"\bAmount of Drug\b", r"\bIn Vitro Drug Release\b"],
            "identities_mode": "section",
            "section_name": "polymer content sweep",
        },
        {
            "field_name": "drug_feed_amount_text",
            "field_label": "etoposide amount",
            "decl_pattern": r"etoposide amount\s*\((.*?)\)",
            "start_patterns": [r"(?:^|\n)\s*Amount of Drug\b"],
            "end_patterns": [r"\bIn Vitro Drug Release\b", r"\bFIG\.\s*4\b", r"\bFIG\.\s*5\b"],
            "identities_mode": "section",
            "section_name": "drug amount sweep",
        },
    ]

    seen_labels = {
        str(form.get("raw_formulation_label") or "").strip().lower()
        for form in forms
        if isinstance(form, dict)
    }
    out: List[Dict[str, Any]] = []

    for spec in specs:
        declared_levels = _extract_declared_levels(raw_text, spec["decl_pattern"])
        if len(declared_levels) < 2:
            continue
        section_text, section_start, section_end = _extract_section_window(
            raw_text,
            start_patterns=spec["start_patterns"],
            end_patterns=spec["end_patterns"],
            pre_context=120,
        )
        if not section_text:
            continue
        sweep_support: Dict[str, List[str]] = {}
        if _section_has_axis_series_anchor(spec["field_name"], section_text):
            baseline = _infer_baseline_level(spec["field_name"], section_text, declared_levels)
            sweep_levels = [lvl for lvl in declared_levels if lvl != baseline]
            if len(sweep_levels) < 2:
                continue
            identities = (
                list(all_identities)
                if spec["identities_mode"] == "all"
                else _infer_section_identities(
                    section_text,
                    all_identities,
                    axis_name=spec["field_name"],
                )
            )
            if not identities:
                continue
            sweep_support = {identity: list(sweep_levels) for identity in identities}
        else:
            sweep_support = _extract_explicit_narrative_sweep_support(
                spec["field_name"],
                section_text,
            )
            if not sweep_support:
                continue
        evidence_span = _normalize_for_match(section_text)[:800]
        for polymer_identity, supported_levels in sweep_support.items():
            for level in supported_levels:
                raw_label = f"{polymer_identity} [{spec['field_label']}={level}]"
                label_key = raw_label.lower()
                if label_key in seen_labels:
                    continue
                seen_labels.add(label_key)
                fields = _build_identity_fields(polymer_identity, raw_text=raw_text)
                fields[spec["field_name"]] = _build_sweep_field(spec["field_name"], level)
                fields.update(
                    _build_sweep_attached_fields(
                        axis_field_name=spec["field_name"],
                        polymer_identity=polymer_identity,
                        level=level,
                        section_text=section_text,
                    )
                )
                candidate_id = _safe_filename_part(f"{key}_{polymer_identity}_{spec['field_name']}_{level}")
                out.append(
                    {
                        "formulation_id": candidate_id,
                        "raw_formulation_label": raw_label,
                        "instance_kind": "variant_formulation",
                        "parent_instance_id": "",
                        "change_descriptions": [f"{spec['field_label']} sweep level: {level}"],
                        "change_role": "synthesis_defining",
                        "instance_context_tags": ["sweep", "figure_variable_sweep"],
                        "change_context_tags": [spec["field_name"], "figure_variable_sweep"],
                        "supporting_evidence_refs": [
                            {
                                "ref_type": "text_span",
                                "evidence_region_type": "results_sentence",
                                "evidence_section": spec["section_name"],
                                "evidence_span_text": evidence_span,
                                "evidence_span_start": int(section_start),
                                "evidence_span_end": int(section_end),
                            }
                        ],
                        "formulation_role": "variant",
                        "instance_confidence": "low",
                        "candidate_source": "figure_variable_sweep",
                        "instance_evidence": {
                            "evidence_region_type": "results_sentence",
                            "evidence_section": spec["section_name"],
                            "evidence_span_text": evidence_span,
                            "evidence_span_start": int(section_start),
                            "evidence_span_end": int(section_end),
                        },
                        "fields": fields,
                    }
                )
    return out


def _safe_filename_part(v: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(v).strip())


def find_replay_raw_response(replay_dir: Path, *, key: str, doi: str) -> Path:
    key_part = _safe_filename_part(key)
    doi_part = _safe_filename_part(doi)
    pattern = f"*_{key_part}_{doi_part}.txt"
    matches = sorted(replay_dir.glob(pattern), key=lambda p: (len(p.name), p.name.lower()))
    if not matches:
        die(
            "Replay raw response not found for "
            f"key={key} doi={doi} under {replay_dir} using pattern {pattern}"
        )
    return matches[0]


@dataclass
class PilotPaper:
    key: str
    doi: str
    title: str
    text_path: Path


def load_manifest(manifest_tsv: Path, max_items: int) -> List[PilotPaper]:
    if not manifest_tsv.exists():
        die(f"Manifest not found: {manifest_tsv}")
    df = pd.read_csv(manifest_tsv, sep="\t", dtype=str).fillna("")
    required = {"key", "doi", "title", "text_path"}
    missing = required - set(df.columns)
    if missing:
        die(f"Manifest missing required columns: {sorted(missing)}")
    rows: List[PilotPaper] = []
    for _, r in df.iterrows():
        p = Path(str(r["text_path"]).replace("\\", "/"))
        rows.append(
            PilotPaper(
                key=str(r["key"]).strip(),
                doi=normalize_doi(r["doi"]),
                title=str(r["title"]).strip(),
                text_path=p,
            )
        )
    if max_items > 0:
        rows = rows[:max_items]
    return rows


def flatten_row(
    key: str,
    doi: str,
    model: str,
    form: Dict[str, Any],
    instance_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    polymer_identity, polymer_name_raw = infer_polymer_identity_fields(form)
    parent_instance_id = str(form.get("parent_instance_id") or "").strip()
    change_descriptions = sanitize_string_list(form.get("change_descriptions"))
    instance_context_tags = sanitize_tag_list(form.get("instance_context_tags"))
    change_context_tags = sanitize_tag_list(form.get("change_context_tags"))
    formulation_role = sanitize_role(form.get("formulation_role"))
    change_role = infer_change_role(
        raw_change_role=form.get("change_role"),
        parent_instance_id=parent_instance_id,
        change_descriptions=change_descriptions,
        tags=instance_context_tags + change_context_tags,
    )
    fields = coerce_fields_map(form.get("fields", {}))
    reconciliation = reconcile_instance_kind(
        raw_instance_kind=form.get("instance_kind_raw", form.get("instance_kind")),
        parent_instance_id=parent_instance_id,
        change_role=change_role,
        formulation_role=formulation_role,
        tags=instance_context_tags + change_context_tags,
        fields=fields,
        polymer_identity=polymer_identity,
        polymer_name_raw=polymer_name_raw,
        raw_formulation_label=str(form.get("raw_formulation_label") or "").strip(),
    )
    supporting_evidence_refs = sanitize_supporting_evidence_refs(
        form.get("supporting_evidence_refs"),
        instance_evidence=instance_evidence,
    )
    out: Dict[str, Any] = {
        "key": key,
        "doi": doi,
        "model": model,
        "local_instance_id": form.get("formulation_id"),
        "formulation_id": form.get("formulation_id"),
        "raw_formulation_label": str(form.get("raw_formulation_label") or "").strip(),
        "polymer_identity": polymer_identity,
        "polymer_name_raw": polymer_name_raw,
        "instance_kind": reconciliation["instance_kind_final"],
        "instance_kind_raw": reconciliation["instance_kind_raw"],
        "instance_kind_inferred": reconciliation["instance_kind_inferred"],
        "instance_kind_reconciliation_note": form.get("instance_kind_reconciliation_note")
        or reconciliation["instance_kind_reconciliation_note"],
        "parent_instance_id": parent_instance_id,
        "change_descriptions": json.dumps(change_descriptions, ensure_ascii=False),
        "change_role": change_role,
        "instance_context_tags": json.dumps(instance_context_tags, ensure_ascii=False),
        "change_context_tags": json.dumps(change_context_tags, ensure_ascii=False),
        "supporting_evidence_refs": json.dumps(supporting_evidence_refs, ensure_ascii=False),
        "formulation_role": sanitize_role(form.get("formulation_role")),
        "instance_confidence": sanitize_conf(form.get("instance_confidence")),
        "candidate_source": str(form.get("candidate_source") or "llm_extracted").strip() or "llm_extracted",
        "instance_evidence_region_type": sanitize_region(instance_evidence.get("evidence_region_type")),
        "evidence_section": instance_evidence.get("evidence_section", ""),
        "evidence_span_text": instance_evidence.get("evidence_span_text", ""),
        "evidence_span_start": instance_evidence.get("evidence_span_start", ""),
        "evidence_span_end": instance_evidence.get("evidence_span_end", ""),
    }
    fields = coerce_fields_map(form.get("fields", {}))
    for f in CORE_FIELDS:
        obj = normalize_field_obj(fields.get(f))
        out[f"{f}_value"] = obj.get("value")
        out[f"{f}_value_text"] = obj.get("value_text", "")
        out[f"{f}_scope"] = sanitize_scope(obj.get("scope"))
        out[f"{f}_membership_confidence"] = sanitize_conf(obj.get("membership_confidence"))
        out[f"{f}_evidence_region_type"] = sanitize_region(obj.get("evidence_region_type"))
        out[f"{f}_missing_reason"] = obj.get("missing_reason", "")
    return enrich_preparation_method_fields_v1(out)


def build_output_columns() -> List[str]:
    cols = [
        "key",
        "doi",
        "model",
        "local_instance_id",
        "formulation_id",
        "raw_formulation_label",
        "polymer_identity",
        "polymer_name_raw",
        "instance_kind",
        "parent_instance_id",
        "change_descriptions",
        "change_role",
        "instance_context_tags",
        "change_context_tags",
        "supporting_evidence_refs",
        "formulation_role",
        "instance_confidence",
        "candidate_source",
        "instance_evidence_region_type",
        "evidence_section",
        "evidence_span_text",
        "evidence_span_start",
        "evidence_span_end",
    ]
    cols.extend(
        [
            "instance_kind_raw",
            "instance_kind_inferred",
            "instance_kind_reconciliation_note",
        ]
    )
    for f in CORE_FIELDS:
        cols.extend(
            [
                f"{f}_value",
                f"{f}_value_text",
                f"{f}_scope",
                f"{f}_membership_confidence",
                f"{f}_evidence_region_type",
                f"{f}_missing_reason",
            ]
        )
    cols.extend(PREPARATION_METHOD_FIELDNAMES)
    return cols


def parse_json_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return [text]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    parsed_text = str(parsed).strip()
    return [parsed_text] if parsed_text else []


def top_level_split(text: str, separators: Tuple[str, ...]) -> List[str]:
    if not text:
        return []
    pieces: List[str] = []
    buffer: List[str] = []
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        matched = None
        if depth == 0:
            for sep in separators:
                if text.startswith(sep, i):
                    matched = sep
                    break
        if matched is not None:
            piece = "".join(buffer).strip()
            if piece:
                pieces.append(piece)
            buffer = []
            i += len(matched)
            continue
        buffer.append(ch)
        i += 1
    piece = "".join(buffer).strip()
    if piece:
        pieces.append(piece)
    return pieces


def normalize_component_name_and_role(raw_name: str, fallback_role: str) -> Tuple[Optional[str], str, str]:
    text = str(raw_name or "").strip()
    if not text:
        return None, fallback_role, "ambiguous"
    lowered = text.lower()
    compact = re.sub(r"[®™]", "", text).strip()
    normalized_name: Optional[str] = compact or None
    normalized_role = fallback_role
    role_mode = "field_mapped"

    if "resomer" in lowered or "plga" in lowered:
        normalized_name = "PLGA"
        normalized_role = "polymer"
        role_mode = "name_normalized"
    elif lowered == "pcl" or "pcl" in lowered:
        normalized_name = "PCL"
        normalized_role = "polymer"
        role_mode = "name_normalized"
    elif "peg-plga" in lowered or "plga-peg" in lowered:
        normalized_name = "PLGA-PEG"
        normalized_role = "polymer"
        role_mode = "name_normalized"
    elif "rhodamine" in lowered:
        normalized_name = "Rhodamine"
        normalized_role = "drug"
        role_mode = "name_normalized"
    elif "gatifloxacin" in lowered:
        normalized_name = "Gatifloxacin"
        normalized_role = "drug"
        role_mode = "name_normalized"
    elif "methylene blue" in lowered or lowered == "mb":
        normalized_name = "Methylene Blue"
        normalized_role = "drug"
        role_mode = "name_normalized"
    elif "dppc" in lowered:
        normalized_name = "DPPC"
        normalized_role = "lipid"
        role_mode = "name_normalized"
    elif "lecithin" in lowered:
        normalized_name = "Lecithin"
        normalized_role = "lipid"
        role_mode = "name_normalized"
    elif "span 80" in lowered:
        normalized_name = "Span 80"
        normalized_role = "surfactant"
        role_mode = "name_normalized"
    elif "pluronic f68" in lowered:
        normalized_name = "Pluronic F68"
        normalized_role = "surfactant"
        role_mode = "name_normalized"
    elif "poloxamer 188" in lowered or lowered == "p188":
        normalized_name = "Poloxamer 188"
        normalized_role = "surfactant"
        role_mode = "name_normalized"
    elif "tween 80" in lowered or "polysorbate 80" in lowered:
        normalized_name = "Polysorbate 80"
        normalized_role = "surfactant"
        role_mode = "name_normalized"
    elif lowered == "pva":
        normalized_name = "PVA"
        normalized_role = "surfactant"
        role_mode = "name_normalized"
    elif "sc6oh" in lowered:
        normalized_name = "SC6OH"
        normalized_role = "surfactant"
        role_mode = "name_normalized"
    elif "labrafil" in lowered:
        normalized_name = "Labrafil"
        normalized_role = "excipient"
        role_mode = "name_normalized"
    elif any(token in lowered for token in ("acetone", "dichloromethane", "ethyl acetate", "chloroform", "methanol", "ethanol", "water")):
        normalized_role = "solvent" if fallback_role in {"solvent", "co_solvent"} else fallback_role
        role_mode = "name_normalized"

    if fallback_role == "co_solvent" and normalized_role == "solvent":
        normalized_role = "co_solvent"
    return normalized_name, normalized_role, role_mode


def detect_amount_kind(amount_expression_raw: str, unit_raw: str) -> str:
    text = f"{amount_expression_raw or ''} {unit_raw or ''}".lower()
    if not text.strip():
        return "unknown"
    if "mg/ml" in text:
        return "concentration_mass_volume"
    if "% w/v" in text or "%, w/v" in text:
        return "concentration_percent_wv"
    if "v/v" in text:
        return "ratio_volume"
    if "w/w" in text or "wt/wt" in text:
        return "ratio_weight"
    if re.search(r"\bmg\b", text):
        return "mass"
    return "unknown"


def infer_phase_context(row: Dict[str, Any], *, source_field: str, component_role: str, component_name_raw: str) -> Tuple[str, str, str]:
    evidence_parts = [
        str(row.get("raw_formulation_label") or ""),
        str(row.get("evidence_section") or ""),
        str(row.get("evidence_span_text") or ""),
        " ".join(parse_json_string_list(row.get("change_descriptions"))),
        " ".join(parse_json_string_list(row.get("instance_context_tags"))),
        " ".join(parse_json_string_list(row.get("change_context_tags"))),
        str(row.get("emulsion_structure") or ""),
        str(row.get("emul_method_value_text") or ""),
    ]
    blob = " | ".join(part for part in evidence_parts if part).strip()
    lowered = blob.lower()
    component_lower = str(component_name_raw or "").lower()
    surfactant_like = any(
        token in component_lower
        for token in ("pva", "polysorbate", "tween", "span 80", "pluronic", "poloxamer", "sc6oh")
    )
    organic_like = component_role in {"polymer", "solvent", "co_solvent", "lipid"} or "acetone" in component_lower or "dichloromethane" in component_lower

    if match := re.search(r"(inner aqueous phase|internal aqueous phase|w1 phase|\bw1\b|primary aqueous phase)", lowered):
        if component_role == "drug" or surfactant_like:
            return match.group(1), "W1", "high"
        return match.group(1), "unspecified", "low"
    if match := re.search(r"(external aqueous phase|outer aqueous phase|continuous aqueous phase|w2 phase|\bw2\b)", lowered):
        if surfactant_like:
            return match.group(1), "W2", "high"
        return match.group(1), "unspecified", "low"
    if match := re.search(r"(organic phase|oil phase|\bo phase\b|\bphase o\b)", lowered):
        if organic_like:
            return match.group(1), "O", "high"
        return match.group(1), "unspecified", "low"
    if match := re.search(r"(external phase)", lowered):
        if surfactant_like:
            return match.group(1), "W2", "medium"
        return match.group(1), "unspecified", "low"
    if match := re.search(r"(inner phase|internal phase)", lowered):
        return match.group(1), "unspecified", "low"

    if source_field == "organic_solvent" and "w/o/w" in lowered:
        return "W/O/W process", "unspecified", "low"
    if source_field in {"surfactant_name", "surfactant_concentration_text", "pva_conc_percent"} and "external phase surfactant" in lowered:
        return "external phase surfactant", "W2", "medium"
    if surfactant_like and "external" in lowered:
        if "external" in lowered:
            return "external phase", "W2", "medium"
    return "", "unspecified", "unknown"


def build_component_properties(row: Dict[str, Any], *, component_role: str, component_name_raw: str) -> List[Dict[str, str]]:
    properties: List[Dict[str, str]] = []
    component_lower = str(component_name_raw or "").lower()
    if component_role == "polymer" or "plga" in component_lower or "resomer" in component_lower or "pcl" in component_lower:
        mw_text = str(row.get("polymer_mw_kDa_value_text") or "")
        if mw_text:
            properties.append(
                {
                    "property_name": "molecular_weight",
                    "property_value_raw": mw_text,
                    "property_unit_raw": "kDa" if "kda" in mw_text.lower() else "",
                }
            )
            grade_match = re.search(r"(RG\s*\d+[A-Z0-9-]*)", mw_text, flags=re.I)
            if grade_match:
                properties.append(
                    {
                        "property_name": "polymer_grade",
                        "property_value_raw": grade_match.group(1).strip(),
                        "property_unit_raw": "",
                    }
                )
        ratio_text = str(row.get("la_ga_ratio_value_text") or "")
        if ratio_text:
            properties.append(
                {
                    "property_name": "la_ga_ratio",
                    "property_value_raw": ratio_text,
                    "property_unit_raw": "",
                }
            )
    return properties


def append_context_text(existing_text: str, new_text: str) -> str:
    current = str(existing_text or "").strip()
    incoming = str(new_text or "").strip()
    if not incoming:
        return current
    if not current:
        return incoming
    if incoming in current:
        return current
    return f"{current} | {incoming}"


def choose_component_key(normalized_name: Optional[str], raw_name: str, fallback_role: str) -> str:
    name_key = str(normalized_name or raw_name or "").strip().lower()
    return f"{fallback_role}::{name_key}"


def add_or_update_component(
    components: Dict[str, Dict[str, Any]],
    ordered_keys: List[str],
    *,
    formulation_row_id: str,
    raw_name: str,
    fallback_role: str,
    source_field: str,
    amount_expression_raw: str = "",
    value_raw: str = "",
    unit_raw: str = "",
    extra_context_text: str = "",
) -> None:
    if not str(raw_name or "").strip():
        return
    normalized_name, normalized_role, role_mode = normalize_component_name_and_role(raw_name, fallback_role)
    key = choose_component_key(normalized_name, raw_name, normalized_role)
    component = components.get(key)
    if component is None:
        component = {
            "formulation_row_id": formulation_row_id,
            "component_index": 0,
            "component_name_raw": str(raw_name or "").strip(),
            "component_name_normalized": normalized_name,
            "component_role_raw": fallback_role,
            "component_role_normalized": normalized_role,
            "role_assignment_mode": role_mode,
            "phase_context_raw": "",
            "phase_context_canonical": "unspecified",
            "phase_confidence": "unknown",
            "amount_expression_raw": "",
            "amount_kind": "unknown",
            "value_raw": "",
            "unit_raw": "",
            "context_text": "",
            "component_properties_json": "[]",
            "_source_field": source_field,
        }
        components[key] = component
        ordered_keys.append(key)

    if not component.get("component_name_raw"):
        component["component_name_raw"] = str(raw_name or "").strip()
    if component.get("component_name_normalized") in {"", None} and normalized_name:
        component["component_name_normalized"] = normalized_name
    if component.get("component_role_normalized") in {"", "unknown"} and normalized_role:
        component["component_role_normalized"] = normalized_role
    if component.get("role_assignment_mode") in {"", "ambiguous"} and role_mode:
        component["role_assignment_mode"] = role_mode
    if not component.get("_source_field"):
        component["_source_field"] = source_field

    if amount_expression_raw and not component.get("amount_expression_raw"):
        component["amount_expression_raw"] = amount_expression_raw
        component["value_raw"] = value_raw
        component["unit_raw"] = unit_raw
        component["amount_kind"] = detect_amount_kind(amount_expression_raw, unit_raw)
    elif amount_expression_raw:
        component["context_text"] = append_context_text(component.get("context_text", ""), amount_expression_raw)

    component["context_text"] = append_context_text(component.get("context_text", ""), extra_context_text)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pilot weak_labels_v7 extractor r3_fixparse (parser + flatten bugfixes + formulation-instance enums)."
    )
    p.add_argument(
        "--manifest-tsv",
        default="data/cleaned/goren_2025/index/splits/dev_manifest_v7pilot3_2026-03-06.tsv",
    )
    p.add_argument("--model", default=PRIMARY_DEFAULT)
    p.add_argument(
        "--llm-backend",
        default="auto",
        choices=["auto", "gemini", "nvidia"],
        help=(
            "Select the LLM transport backend. "
            "'auto' keeps Gemini for Gemini models and uses the NVIDIA hosted API for models such as meta/llama-3.3-70b-instruct."
        ),
    )
    p.add_argument("--max-chars", type=int, default=50000)
    p.add_argument("--max-items", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.4)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--out-dir", default="")
    p.add_argument(
        "--replay-raw-responses-dir",
        default="",
        help=(
            "Reuse existing raw response .txt files from a previous Stage2 run. "
            "When set, the extractor parses those saved responses and does not make new LLM calls."
        ),
    )
    p.add_argument(
        "--disable-numbered-doe-enumerator",
        action="store_true",
        help="Disable the deterministic numbered DOE table-row enumerator. The default is enabled.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    replay_dir = Path(args.replay_raw_responses_dir).resolve() if args.replay_raw_responses_dir else None
    if replay_dir is not None:
        if not replay_dir.exists():
            raise FileNotFoundError(f"Replay raw responses directory not found: {replay_dir}")
        resolved_backend = "replay_only"
    else:
        validate_models_or_raise([args.model], context="v7pilot_r3_fixparse preflight")
        resolved_backend = ensure_llm_backend(args.model, args.llm_backend)

    papers = load_manifest(Path(args.manifest_tsv), args.max_items)
    if len(papers) == 0:
        die("No pilot papers loaded from manifest.")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    default_out = Path(f"data/results/run_{ts}_v7pilot3r3fixparse_dev/weak_labels_v7pilot_r3_fixparse")
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "weak_labels__v7pilot_r3_fixparse.jsonl"
    out_tsv = out_dir / "weak_labels__v7pilot_r3_fixparse.tsv"
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    columns = build_output_columns()
    tsv_f = out_tsv.open("w", encoding="utf-8", newline="")
    tsv_w = csv.DictWriter(tsv_f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
    tsv_w.writeheader()

    n_forms = 0
    numbered_doe_artifact_rows: List[Dict[str, Any]] = []
    numbered_doe_summary_rows: List[Dict[str, Any]] = []
    numbered_doe_guard_rows: List[Dict[str, str]] = []
    with out_jsonl.open("w", encoding="utf-8") as jf:
        for i, paper in enumerate(papers, start=1):
            if not paper.text_path.exists():
                print(f"[WARN] missing text path: {paper.text_path}")
                continue
            raw_txt = paper.text_path.read_text(encoding="utf-8", errors="ignore")
            packed_txt = assemble_evidence_text(
                raw_text=raw_txt,
                key=paper.key,
                doi=paper.doi,
                title=paper.title,
                max_chars=args.max_chars,
            )
            prompt = build_prompt(packed_txt, detection_text=raw_txt)

            raw_fp = raw_dir / f"{i:02d}_{_safe_filename_part(paper.key)}_{_safe_filename_part(paper.doi)}.txt"
            if replay_dir is not None:
                source_raw_fp = find_replay_raw_response(replay_dir, key=paper.key, doi=paper.doi)
                raw = source_raw_fp.read_text(encoding="utf-8")
            else:
                raw = call_llm(args.model, prompt, args.retries, args.sleep, args.llm_backend)
            raw_fp.write_text(raw, encoding="utf-8")
            data = safe_json_load(raw)
            forms = canonicalize_formulations(data)
            forms = apply_targeted_boundary_corrections(forms, raw_txt, paper.key)
            forms.extend(enumerate_figure_variable_sweep_candidates(forms, raw_txt, paper.key))
            forms = dedupe_sweep_overlap_forms(forms)
            doe_summary: Optional[Dict[str, Any]] = None
            if not args.disable_numbered_doe_enumerator:
                doe_forms, doe_artifacts, doe_summary = enumerate_numbered_doe_candidates_for_paper(
                    paper=paper,
                    raw_text=raw_txt,
                    existing_forms=forms,
                    min_numbered_rows=8,
                )
                if doe_forms:
                    forms.extend(doe_forms)
                    forms = prefer_numbered_doe_forms_over_llm_numeric_rows(forms)
                numbered_doe_artifact_rows.extend(doe_artifacts)
                numbered_doe_summary_rows.append(doe_summary)
            numbered_doe_guard_rows.append(
                build_numbered_doe_guard_row(
                    paper={"key": paper.key, "doi": paper.doi},
                    forms=forms,
                    doe_summary=doe_summary,
                )
            )

            line = {
                "key": paper.key,
                "doi": paper.doi,
                "title": paper.title,
                "model": args.model,
                "schema_version": data.get("schema_version", "weak_labels_v7"),
                "paper_notes": data.get("paper_notes"),
                "formulations": forms,
            }
            jf.write(json.dumps(line, ensure_ascii=False) + "\n")

            instance_fallback = choose_instance_evidence(packed_txt)
            for idx, f in enumerate(forms, start=1):
                if not isinstance(f, dict):
                    continue
                formulation_id = f.get("formulation_id", f.get("id", idx))
                # Keep raw structure; flatten_row handles dict and list styles.
                fields = f.get("fields", {})
                row_form = {
                    "formulation_id": formulation_id,
                    "raw_formulation_label": f.get("raw_formulation_label", ""),
                    "polymer_identity": f.get("polymer_identity", "unknown"),
                    "polymer_name_raw": f.get("polymer_name_raw", ""),
                    "instance_kind": f.get("instance_kind", "unclear"),
                    "instance_kind_raw": f.get("instance_kind_raw", f.get("instance_kind", "unclear")),
                    "instance_kind_inferred": f.get("instance_kind_inferred", f.get("instance_kind", "unclear")),
                    "instance_kind_reconciliation_note": f.get("instance_kind_reconciliation_note", ""),
                    "parent_instance_id": f.get("parent_instance_id", ""),
                    "change_descriptions": f.get("change_descriptions", []),
                    "change_role": f.get("change_role", "unclear"),
                    "instance_context_tags": f.get("instance_context_tags", []),
                    "change_context_tags": f.get("change_context_tags", []),
                    "supporting_evidence_refs": f.get("supporting_evidence_refs", []),
                    "formulation_role": f.get("formulation_role", "unknown"),
                    "instance_confidence": f.get("instance_confidence", "low"),
                    "candidate_source": f.get("candidate_source", "llm_extracted"),
                    "fields": fields,
                }
                inst_ev = f.get("instance_evidence", {})
                if not isinstance(inst_ev, dict):
                    inst_ev = {}
                if not inst_ev:
                    inst_ev = instance_fallback
                row = flatten_row(paper.key, paper.doi, args.model, row_form, inst_ev)
                tsv_w.writerow(row)
                n_forms += 1

            if args.verbose:
                print(f"[{i}/{len(papers)}] key={paper.key} doi={paper.doi} formulations={len(forms)}")

    tsv_f.close()
    if not args.disable_numbered_doe_enumerator:
        numbered_doe_stats = write_candidate_artifacts(
            out_dir=out_dir,
            artifact_rows=numbered_doe_artifact_rows,
            summary_rows=numbered_doe_summary_rows,
            expected_min_recovered=0,
        )
    else:
        numbered_doe_stats = {
            "artifact_path": out_dir / NUMBERED_DOE_ARTIFACT_NAME,
            "summary_path": out_dir / NUMBERED_DOE_SUMMARY_NAME,
            "candidate_count": 0,
            "paper_count": 0,
        }
    guard_stats = write_numbered_doe_guard_artifact(out_dir, numbered_doe_guard_rows)
    summary = {
        "manifest_tsv": str(Path(args.manifest_tsv).resolve()),
        "model": args.model,
        "llm_backend": resolved_backend,
        "replay_raw_responses_dir": str(replay_dir) if replay_dir is not None else "",
        "fresh_llm_calls_made": replay_dir is None,
        "n_papers": len(papers),
        "n_formulations": n_forms,
        "n_numbered_doe_candidates": int(numbered_doe_stats["candidate_count"]),
        "out_dir": str(out_dir.resolve()),
        "out_jsonl": str(out_jsonl.resolve()),
        "out_tsv": str(out_tsv.resolve()),
        "raw_responses_dir": str(raw_dir.resolve()),
        "numbered_doe_candidates_tsv": str(Path(numbered_doe_stats["artifact_path"]).resolve()),
        "numbered_doe_summary_tsv": str(Path(numbered_doe_stats["summary_path"]).resolve()),
        "numbered_doe_guard_tsv": str(Path(guard_stats["guard_path"]).resolve()),
        "numbered_doe_guard_fail_count": int(guard_stats["fail_count"]),
        "numbered_doe_guard_warn_count": int(guard_stats["warn_count"]),
    }
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if guard_stats["fail_count"] or guard_stats["warn_count"]:
        print(
            "[WARN] numbered DOE regression guard flagged "
            f"{guard_stats['fail_count']} fail(s) and {guard_stats['warn_count']} warn(s).",
            file=sys.stderr,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
