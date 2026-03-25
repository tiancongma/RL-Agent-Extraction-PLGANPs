#!/usr/bin/env python3
from __future__ import annotations

# ARCHIVED: Stage2.5 exploratory component-shadow pipeline.
# Not part of active mainline as of 2026-03-25.

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from src.utils.active_data_source import (
        build_artifact_metadata,
        resolve_artifact_path,
        resolve_run_context,
        write_artifact_metadata_json,
    )
    from src.utils.paths import DATA_CLEANED_INDEX_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.active_data_source import (
        build_artifact_metadata,
        resolve_artifact_path,
        resolve_run_context,
        write_artifact_metadata_json,
    )
    from src.utils.paths import DATA_CLEANED_INDEX_DIR, PROJECT_ROOT


ALLOWED_ANCHOR_STATUS = {
    "exact_match",
    "multi_match_resolved",
    "failed",
    "exact_match_but_out_of_scope",
}
ALLOWED_BINDING_MODES = {"row_direct", "row_inferred_local", "shared_context", "unbound"}
PREPARATION_VERBS = {
    "added",
    "dissolved",
    "emulsified",
    "mixed",
    "prepared",
    "dispersed",
    "stirred",
    "homogenized",
    "sonicated",
    "evaporated",
    "poured",
    "transferred",
}
PHASE_CUES = {
    "aqueous",
    "organic",
    "oil phase",
    "water phase",
    "external phase",
    "internal phase",
    "inner aqueous",
    "outer aqueous",
    "w/o/w",
    "o/w",
}
PROPERTY_CUES = {
    "kda",
    "molecular weight",
    "mw",
    "la/ga",
    "50:50",
    "75:25",
    "85:15",
    "resomer",
    "inherent viscosity",
    "end group",
    "grade",
    "supplier",
}
BOILERPLATE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"creative commons",
        r"open access",
        r"terms of use",
        r"all rights reserved",
        r"received .* accepted",
        r"journal homepage",
        r"previous article in issue",
        r"next article in issue",
        r"submit your manuscript",
        r"doi:",
        r"references",
        r"acknowledg",
    ]
]
AMOUNT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg(?:/mL)?|g|kg|mL|uL|% ?w/v|% ?v/v|%|kDa)\b",
    re.IGNORECASE,
)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_CANDIDATES_PER_ROW = 12


@dataclass(frozen=True)
class SentenceSpan:
    sentence_index: int
    char_start: int
    char_end: int
    raw_text_exact: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage2.5A v0 text-only exact-anchored evidence packs from frozen Stage2 rows."
    )
    parser.add_argument("--run-dir", default="", help="Optional authoritative run directory. Overrides ACTIVE_RUN.json.")
    parser.add_argument("--run-id", default="", help="Optional compatibility alias for resolving a top-level results run.")
    parser.add_argument("--stage2-tsv", default="", help="Optional explicit frozen Stage2 TSV. Default: active run weak-label TSV.")
    parser.add_argument("--scope-manifest-tsv", default="", help="Optional explicit scope manifest TSV. Default: ACTIVE_RUN.json scope manifest.")
    parser.add_argument("--paper-keys-file", default="", help="Optional one-key-per-line subset filter.")
    parser.add_argument("--max-rows-per-key", type=int, default=0, help="Optional positive cap for formulation rows per paper key after key filtering.")
    parser.add_argument("--out-dir", required=True, help="Run-scoped output directory for the shadow evidence-pack artifacts.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\ufeff", " ")).strip()


def ascii_preview(value: str, limit: int = 180) -> str:
    return normalize_text(value).encode("ascii", "ignore").decode("ascii")[:limit]


def repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def stable_formulation_row_id(row: dict[str, str]) -> str:
    key = normalize_text(row.get("key"))
    local_id = normalize_text(row.get("local_instance_id"))
    formulation_id = normalize_text(row.get("formulation_id"))
    local_or_formulation = local_id or formulation_id
    if key and local_or_formulation:
        return f"{key}__{local_or_formulation}"
    return local_or_formulation or key


def load_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: str(row.get(key, "")) for key in fieldnames})


def read_keys_file(path: Path) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        key = raw_line.strip()
        if not key or key.startswith("#") or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def resolve_repo_path(raw_path: str, base_dir: Path) -> Path:
    candidate = Path(str(raw_path or "").strip())
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def load_text_path_map(scope_manifest_path: Path) -> dict[str, Path]:
    rows = load_tsv_rows(scope_manifest_path)
    mapping: dict[str, Path] = {}
    for row in rows:
        key = normalize_text(row.get("key") or row.get("paper_id"))
        text_path = normalize_text(row.get("text_path"))
        if not key or not text_path:
            continue
        resolved = resolve_repo_path(text_path, PROJECT_ROOT)
        if resolved.exists():
            mapping[key] = resolved
    if mapping:
        return mapping

    fallback_path = DATA_CLEANED_INDEX_DIR / "key2txt.tsv"
    if not fallback_path.exists():
        return {}
    for line in fallback_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        key = normalize_text(parts[0])
        rel = normalize_text(parts[1])
        if not key or not rel:
            continue
        resolved = resolve_repo_path(rel, PROJECT_ROOT)
        if resolved.exists():
            mapping[key] = resolved
    return mapping


def trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


def build_sentence_spans(text: str) -> list[SentenceSpan]:
    spans: list[SentenceSpan] = []
    last = 0
    sentence_index = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        trimmed = trim_span(text, last, match.start())
        if trimmed is not None:
            spans.append(
                SentenceSpan(
                    sentence_index=sentence_index,
                    char_start=trimmed[0],
                    char_end=trimmed[1],
                    raw_text_exact=text[trimmed[0]:trimmed[1]],
                )
            )
            sentence_index += 1
        last = match.end()
    trimmed = trim_span(text, last, len(text))
    if trimmed is not None:
        spans.append(
            SentenceSpan(
                sentence_index=sentence_index,
                char_start=trimmed[0],
                char_end=trimmed[1],
                raw_text_exact=text[trimmed[0]:trimmed[1]],
            )
        )
    return spans


def get_context(text: str, start: int, end: int, width: int = 120) -> tuple[str, str]:
    before = text[max(0, start - width):start]
    after = text[end:min(len(text), end + width)]
    return normalize_text(before), normalize_text(after)


def count_exact_matches(text: str, raw_text_exact: str) -> list[int]:
    if not raw_text_exact:
        return []
    starts: list[int] = []
    cursor = 0
    while True:
        idx = text.find(raw_text_exact, cursor)
        if idx < 0:
            break
        starts.append(idx)
        cursor = idx + 1
    return starts


def is_trivial_text(raw_text_exact: str) -> bool:
    compact = normalize_text(raw_text_exact)
    if not compact:
        return True
    if not re.search(r"[A-Za-z0-9]", compact):
        return True
    return len(compact) < 4


def is_boilerplate(raw_text_exact: str) -> bool:
    compact = normalize_text(raw_text_exact)
    if not compact:
        return True
    return any(pattern.search(compact) for pattern in BOILERPLATE_PATTERNS)


def looks_like_method_span(raw_text_exact: str) -> bool:
    lowered = normalize_text(raw_text_exact).lower()
    return any(verb in lowered for verb in PREPARATION_VERBS)


def normalize_for_match(value: str) -> str:
    return normalize_text(value).lower()


def hint_regex(value: str) -> re.Pattern[str] | None:
    compact = normalize_text(value)
    if len(compact) < 2:
        return None
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(compact)}(?![A-Za-z0-9])", re.IGNORECASE)


def extract_hint_phrases(value: str) -> list[str]:
    compact = normalize_text(value)
    if not compact:
        return []
    phrases = [compact]
    if "(" in compact and ")" in compact:
        phrases.extend(normalize_text(item) for item in re.findall(r"\(([^)]+)\)", compact))
        phrases.append(normalize_text(re.sub(r"\([^)]*\)", " ", compact)))
    for piece in re.split(r"[;,]| and | or ", compact):
        piece = normalize_text(piece)
        if piece:
            phrases.append(piece)
    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        key = normalize_for_match(phrase)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(phrase)
    return deduped


def collect_row_hints(row: dict[str, str]) -> dict[str, list[str]]:
    direct_fields = [row.get("formulation_id", ""), row.get("raw_formulation_label", "")]
    material_fields = [
        row.get("polymer_identity", ""),
        row.get("polymer_name_raw", ""),
        row.get("surfactant_name_value", ""),
        row.get("surfactant_name_value_text", ""),
        row.get("organic_solvent_value", ""),
        row.get("organic_solvent_value_text", ""),
        row.get("drug_name_value", ""),
        row.get("drug_name_value_text", ""),
        row.get("aqueous_phase_value", ""),
        row.get("aqueous_phase_value_text", ""),
    ]
    property_fields = [row.get("plga_mw_kDa_value_text", ""), row.get("la_ga_ratio_value_text", "")]
    amount_fields = [
        row.get("plga_mass_mg_value_text", ""),
        row.get("surfactant_concentration_text_value_text", ""),
        row.get("pva_conc_percent_value_text", ""),
        row.get("drug_feed_amount_text_value_text", ""),
    ]
    hints = {"row_direct": [], "material": [], "property": [], "amount": []}
    for target, values in [
        ("row_direct", direct_fields),
        ("material", material_fields),
        ("property", property_fields),
        ("amount", amount_fields),
    ]:
        seen: set[str] = set()
        for value in values:
            for phrase in extract_hint_phrases(value):
                match_key = normalize_for_match(phrase)
                if not match_key or match_key in seen:
                    continue
                seen.add(match_key)
                hints[target].append(phrase)
    return hints


def hint_overlap_count(sentence_text: str, values: list[str]) -> int:
    count = 0
    for value in values:
        regex = hint_regex(value)
        if regex is not None and regex.search(sentence_text):
            count += 1
    return count


def collect_field_hints(sentence_text: str, row_hints: dict[str, list[str]]) -> list[str]:
    lowered = normalize_for_match(sentence_text)
    field_hints: set[str] = set()
    if hint_overlap_count(sentence_text, row_hints["material"]) > 0:
        field_hints.add("component_name")
    if hint_overlap_count(sentence_text, row_hints["amount"]) > 0 or AMOUNT_PATTERN.search(sentence_text):
        field_hints.add("amount")
    if any(cue in lowered for cue in PHASE_CUES):
        field_hints.add("phase")
    if hint_overlap_count(sentence_text, row_hints["property"]) > 0 or any(cue in lowered for cue in PROPERTY_CUES):
        field_hints.add("property")
    if not field_hints:
        field_hints.add("unknown")
    return sorted(field_hints)


def has_row_direct_match(sentence_text: str, row_hints: dict[str, list[str]]) -> bool:
    return hint_overlap_count(sentence_text, row_hints["row_direct"]) > 0


def is_global_row_direct_phrase(phrase: str) -> bool:
    compact = normalize_text(phrase)
    if not compact:
        return False
    if re.fullmatch(r"F\d+(?:\.\d+)?", compact, re.IGNORECASE):
        return False
    return len(compact) >= 4


def parse_int(value: str) -> int | None:
    try:
        return int(str(value or "").strip())
    except Exception:
        return None


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def build_locator(span: SentenceSpan) -> dict[str, Any]:
    return {
        "asset_type": "clean_text",
        "sentence_index": span.sentence_index,
        "char_start": span.char_start,
        "char_end": span.char_end,
    }


def make_dedupe_key(row_id: str, source_path: Path, span: SentenceSpan) -> str:
    raw = "|".join(
        [
            row_id,
            str(source_path.resolve()),
            str(span.char_start),
            str(span.char_end),
            normalize_for_match(span.raw_text_exact),
        ]
    )
    return re.sub(r"\s+", " ", raw).strip()


def build_evidence_unit(
    *,
    evidence_id: str,
    row_id: str,
    source_path: Path,
    span: SentenceSpan,
    text: str,
    field_hints: list[str],
    binding_mode: str,
    anchor_status: str,
    is_valid_anchor: bool,
    is_empty: bool,
    rejection_reason: str,
) -> dict[str, Any]:
    context_before, context_after = get_context(text, span.char_start, span.char_end)
    source_type = "method_span" if looks_like_method_span(span.raw_text_exact) else "sentence_span"
    return {
        "evidence_id": evidence_id,
        "formulation_row_id": row_id,
        "source_type": source_type,
        "source_path": str(source_path.resolve()),
        "source_locator": build_locator(span),
        "raw_text_exact": span.raw_text_exact,
        "normalized_text": normalize_text(span.raw_text_exact),
        "anchor_status": anchor_status,
        "context_before": context_before,
        "context_after": context_after,
        "field_hints": field_hints,
        "dedupe_key": make_dedupe_key(row_id, source_path, span),
        "is_valid_anchor": bool(is_valid_anchor),
        "is_empty": bool(is_empty),
        "binding_mode": binding_mode,
        "rejection_reason": rejection_reason,
    }


def classify_binding_mode(
    *,
    sentence_text: str,
    row_hints: dict[str, list[str]],
    in_local_window: bool,
    field_hints: list[str],
) -> str:
    if has_row_direct_match(sentence_text, row_hints):
        return "row_direct"
    if in_local_window:
        hint_count = len([hint for hint in field_hints if hint != "unknown"])
        material_hits = hint_overlap_count(sentence_text, row_hints["material"])
        amount_hits = hint_overlap_count(sentence_text, row_hints["amount"])
        lowered = normalize_for_match(sentence_text)
        if hint_count >= 2:
            return "row_inferred_local"
        if material_hits >= 1 and amount_hits >= 1:
            return "row_inferred_local"
        if material_hits >= 1 and any(verb in lowered for verb in PREPARATION_VERBS):
            return "row_inferred_local"
        return "shared_context"
    if any(verb in normalize_for_match(sentence_text) for verb in PREPARATION_VERBS):
        return "shared_context"
    return "unbound"


def select_candidate_spans(
    *,
    spans: list[SentenceSpan],
    row: dict[str, str],
    row_hints: dict[str, list[str]],
) -> list[SentenceSpan]:
    scored: list[tuple[int, int, SentenceSpan]] = []
    local_start = parse_int(row.get("evidence_span_start"))
    local_end = parse_int(row.get("evidence_span_end"))
    global_direct_phrases = [phrase for phrase in row_hints["row_direct"] if is_global_row_direct_phrase(phrase)]
    for span in spans:
        sentence_text = span.raw_text_exact
        in_local = (
            local_start is not None
            and local_end is not None
            and overlap(span.char_start, span.char_end, local_start, local_end)
        )
        material_hits = hint_overlap_count(sentence_text, row_hints["material"])
        amount_hits = hint_overlap_count(sentence_text, row_hints["amount"])
        property_hits = hint_overlap_count(sentence_text, row_hints["property"])
        lowered = normalize_for_match(sentence_text)
        prep_hit = any(verb in lowered for verb in PREPARATION_VERBS)
        phase_hit = any(cue in lowered for cue in PHASE_CUES)
        global_direct_hit = any(
            regex is not None and regex.search(sentence_text)
            for regex in [hint_regex(phrase) for phrase in global_direct_phrases]
        )

        score = 0
        if in_local and (material_hits or amount_hits or property_hits or prep_hit or phase_hit):
            score += 6
        if global_direct_hit:
            score += 5
        if material_hits and amount_hits:
            score += 4
        if prep_hit and (material_hits or amount_hits):
            score += 3
        if property_hits:
            score += 2
        if phase_hit:
            score += 1
        if score <= 0:
            continue
        scored.append((score, span.char_start, span))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected: dict[tuple[int, int], SentenceSpan] = {}
    for _, _, span in scored[:MAX_CANDIDATES_PER_ROW]:
        selected[(span.char_start, span.char_end)] = span
    return [selected[key] for key in sorted(selected)]


def rank_units(units: list[dict[str, Any]], preferred_mode: str, limit: int) -> list[dict[str, Any]]:
    def sort_key(unit: dict[str, Any]) -> tuple[int, int, int]:
        mode_rank = 0 if unit["binding_mode"] == preferred_mode else 1
        hint_rank = -len([item for item in unit["field_hints"] if item != "unknown"])
        length_rank = len(unit["normalized_text"])
        return mode_rank, hint_rank, length_rank

    return sorted(units, key=sort_key)[:limit]


def process_row(
    *,
    row: dict[str, str],
    text_path: Path,
    text_cache: dict[Path, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter]:
    row_id = stable_formulation_row_id(row)
    if text_path not in text_cache:
        text_cache[text_path] = text_path.read_text(encoding="utf-8", errors="ignore")
    text = text_cache[text_path]
    spans = build_sentence_spans(text)
    row_hints = collect_row_hints(row)
    local_start = parse_int(row.get("evidence_span_start"))
    local_end = parse_int(row.get("evidence_span_end"))

    strict_units: list[dict[str, Any]] = []
    supporting_units: list[dict[str, Any]] = []
    rejected_units: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    dedupe_status: dict[str, str] = {}
    counters: Counter = Counter()

    candidate_spans = select_candidate_spans(spans=spans, row=row, row_hints=row_hints)
    candidate_spans.sort(key=lambda item: (item.char_start, item.char_end))

    evidence_counter = 1
    for span in candidate_spans:
        sentence_text = span.raw_text_exact
        field_hints = collect_field_hints(sentence_text, row_hints)
        lowered = normalize_for_match(sentence_text)
        prep_hit = any(verb in lowered for verb in PREPARATION_VERBS)
        in_local_window = (
            local_start is not None
            and local_end is not None
            and overlap(span.char_start, span.char_end, local_start, local_end)
        )
        match_positions = count_exact_matches(text, sentence_text)
        if is_trivial_text(sentence_text):
            anchor_status = "failed"
            is_valid_anchor = False
            is_empty = True
            binding_mode = "unbound"
            rejection_reason = "empty_or_trivial_span"
        elif not match_positions:
            anchor_status = "failed"
            is_valid_anchor = False
            is_empty = False
            binding_mode = "unbound"
            rejection_reason = "failed_exact_anchor"
        else:
            is_valid_anchor = True
            is_empty = False
            if len(match_positions) == 1:
                anchor_status = "exact_match"
            elif span.char_start in match_positions:
                anchor_status = "multi_match_resolved"
            else:
                anchor_status = "failed"
                is_valid_anchor = False
            binding_mode = classify_binding_mode(
                sentence_text=sentence_text,
                row_hints=row_hints,
                in_local_window=in_local_window,
                field_hints=field_hints,
            )
            rejection_reason = ""

        if is_valid_anchor and not in_local_window and has_row_direct_match(sentence_text, row_hints):
            strong_row_direct_signal = prep_hit or (
                "component_name" in field_hints and "amount" in field_hints
            )
            if not strong_row_direct_signal:
                anchor_status = "exact_match_but_out_of_scope"
                rejection_reason = "out_of_scope_row_direct_span"

        if is_valid_anchor and is_boilerplate(sentence_text):
            rejection_reason = "boilerplate_text"

        unit = build_evidence_unit(
            evidence_id=f"{row_id}__ev_{evidence_counter:03d}",
            row_id=row_id,
            source_path=text_path,
            span=span,
            text=text,
            field_hints=field_hints,
            binding_mode=binding_mode if binding_mode in ALLOWED_BINDING_MODES else "unbound",
            anchor_status=anchor_status if anchor_status in ALLOWED_ANCHOR_STATUS else "failed",
            is_valid_anchor=is_valid_anchor,
            is_empty=is_empty,
            rejection_reason=rejection_reason,
        )
        evidence_counter += 1

        dedupe_key = unit["dedupe_key"]
        if dedupe_key in dedupe_status:
            unit["rejection_reason"] = "duplicate_exact"
            rejected_units.append(unit)
            counters["rejected_units"] += 1
            counters["duplicate_exact"] += 1
        elif unit["anchor_status"] == "failed":
            rejected_units.append(unit)
            counters["rejected_units"] += 1
            counters["failed_anchors"] += 1
        elif unit["rejection_reason"] == "boilerplate_text":
            rejected_units.append(unit)
            counters["rejected_units"] += 1
            counters["boilerplate_rejections"] += 1
        elif unit["anchor_status"] == "exact_match_but_out_of_scope":
            rejected_units.append(unit)
            counters["rejected_units"] += 1
            counters["out_of_scope_spans"] += 1
        elif unit["binding_mode"] in {"row_direct", "row_inferred_local"}:
            strict_units.append(unit)
            dedupe_status[dedupe_key] = "strict"
            counters["strict_units"] += 1
        elif unit["binding_mode"] == "shared_context":
            supporting_units.append(unit)
            dedupe_status[dedupe_key] = "supporting"
            counters["supporting_units"] += 1
        else:
            unit["rejection_reason"] = "unbound_span"
            rejected_units.append(unit)
            counters["rejected_units"] += 1
            counters["unbound_spans"] += 1

    strict_units = rank_units(strict_units, preferred_mode="row_direct", limit=3)
    supporting_units = rank_units(supporting_units, preferred_mode="shared_context", limit=3)

    for bucket_name, units in [
        ("strict", strict_units),
        ("supporting", supporting_units),
        ("rejected", rejected_units),
    ]:
        for unit in units:
            audit_rows.append(
                {
                    "formulation_row_id": row_id,
                    "paper_key": row.get("key", ""),
                    "evidence_id": unit["evidence_id"],
                    "bucket": bucket_name,
                    "source_type": unit["source_type"],
                    "binding_mode": unit["binding_mode"],
                    "anchor_status": unit["anchor_status"],
                    "field_hints": ",".join(unit["field_hints"]),
                    "raw_text_exact": unit["raw_text_exact"],
                    "rejection_reason": unit["rejection_reason"],
                    "context_preview": ascii_preview(
                        f"{unit['context_before']} >>> {unit['raw_text_exact']} <<< {unit['context_after']}",
                        limit=260,
                    ),
                }
            )

    pack = {
        "formulation_row_id": row_id,
        "paper_key": row.get("key", ""),
        "strict_evidence_units": strict_units,
        "supporting_evidence_units": supporting_units,
        "rejected_evidence_units": rejected_units,
    }
    return pack, audit_rows, counters


def summarize_examples(packs: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    buckets: dict[str, list[dict[str, str]]] = {"strict": [], "supporting": [], "rejected": []}
    for pack in packs:
        row_id = pack["formulation_row_id"]
        paper_key = pack["paper_key"]
        for bucket_name, units in [
            ("strict", pack["strict_evidence_units"]),
            ("supporting", pack["supporting_evidence_units"]),
            ("rejected", pack["rejected_evidence_units"]),
        ]:
            for unit in units:
                if len(buckets[bucket_name]) >= 5:
                    break
                buckets[bucket_name].append(
                    {
                        "paper_key": paper_key,
                        "formulation_row_id": row_id,
                        "binding_mode": unit["binding_mode"],
                        "anchor_status": unit["anchor_status"],
                        "rejection_reason": unit["rejection_reason"],
                        "raw_text_exact_preview": ascii_preview(unit["raw_text_exact"], limit=220),
                    }
                )
    return buckets


def build_validation_summary_md(
    *,
    stage2_tsv: Path,
    scope_manifest_tsv: Path,
    packs_jsonl: Path,
    audit_tsv: Path,
    summary_json: Path,
    selected_rows: list[dict[str, str]],
    counts: Counter,
    examples: dict[str, list[dict[str, str]]],
) -> str:
    selected_keys = sorted({row.get("key", "") for row in selected_rows})
    lines = [
        "# Stage2.5A Text Evidence Pack v0 Validation Summary",
        "",
        "## Status",
        "",
        "- `diagnostic-only, not benchmark-valid final output`",
        "- Scope: text evidence only; no table binding; no Stage2/Stage3/Stage5 behavior changes.",
        "",
        "## Inputs",
        "",
        f"- Stage2 TSV: `{repo_rel(stage2_tsv)}`",
        f"- Scope manifest: `{repo_rel(scope_manifest_tsv)}`",
        f"- Selected paper keys: `{', '.join(selected_keys)}`",
        "",
        "## Counts",
        "",
        f"- formulation rows processed: `{counts['rows_processed']}`",
        f"- strict units: `{counts['strict_units']}`",
        f"- supporting units: `{counts['supporting_units']}`",
        f"- rejected units: `{counts['rejected_units']}`",
        f"- failed anchors: `{counts['failed_anchors']}`",
        f"- out-of-scope spans: `{counts['out_of_scope_spans']}`",
        "",
        "## Representative Examples",
        "",
    ]
    for bucket_name in ["strict", "supporting", "rejected"]:
        lines.append(f"### {bucket_name.title()}")
        lines.append("")
        if not examples[bucket_name]:
            lines.append("- none")
            lines.append("")
            continue
        for example in examples[bucket_name]:
            lines.append(
                "- "
                f"`{example['paper_key']}` / `{example['formulation_row_id']}` / "
                f"`{example['binding_mode']}` / `{example['anchor_status']}` / "
                f"`{example['rejection_reason'] or 'ok'}` / "
                f"{example['raw_text_exact_preview']}"
            )
        lines.append("")
    lines.extend(
        [
            "## Output Artifacts",
            "",
            f"- packs JSONL: `{repo_rel(packs_jsonl)}`",
            f"- audit TSV: `{repo_rel(audit_tsv)}`",
            f"- summary JSON: `{repo_rel(summary_json)}`",
            "",
            "## Notes",
            "",
            "- Strict evidence is limited to exact-anchored text spans classified as `row_direct` or `row_inferred_local`.",
            "- Shared method-like text is preserved only as supporting evidence in v0.",
            "- Out-of-scope exact anchors are rejected rather than promoted into strict evidence.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    explicit_run_dir = Path(args.run_dir).resolve() if normalize_text(args.run_dir) else None
    run_context = resolve_run_context(
        explicit_run_dir=explicit_run_dir,
        explicit_run_id=normalize_text(args.run_id),
    )
    stage2_tsv = resolve_artifact_path(
        explicit_path=Path(args.stage2_tsv).resolve() if normalize_text(args.stage2_tsv) else None,
        run_context=run_context,
        pointer_key="stage2_candidate_tsv",
        canonical_relative="weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv",
        required=True,
    )
    scope_manifest_tsv = resolve_artifact_path(
        explicit_path=Path(args.scope_manifest_tsv).resolve() if normalize_text(args.scope_manifest_tsv) else None,
        run_context=run_context,
        pointer_key="scope_manifest_tsv",
        required=True,
    )
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = out_dir / "outputs"
    analysis_dir = out_dir / "analysis"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"resolved_source_run_dir\t{run_context['run_dir']}")
    print(f"resolved_source_run_id\t{run_context['run_id']}")
    print(f"source_stage2_tsv\t{stage2_tsv}")
    print(f"source_scope_manifest_tsv\t{scope_manifest_tsv}")

    subset_keys: list[str] = []
    if normalize_text(args.paper_keys_file):
        subset_keys = read_keys_file(Path(args.paper_keys_file).resolve())
    rows = load_tsv_rows(stage2_tsv)
    if subset_keys:
        key_set = set(subset_keys)
        rows = [row for row in rows if normalize_text(row.get("key")) in key_set]
    if args.max_rows_per_key > 0:
        kept: list[dict[str, str]] = []
        seen_by_key: dict[str, int] = defaultdict(int)
        for row in rows:
            key = normalize_text(row.get("key"))
            if seen_by_key[key] >= args.max_rows_per_key:
                continue
            kept.append(row)
            seen_by_key[key] += 1
        rows = kept

    text_path_map = load_text_path_map(scope_manifest_tsv)
    text_cache: dict[Path, str] = {}
    packs: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    counts: Counter = Counter()

    for row in rows:
        key = normalize_text(row.get("key"))
        row_id = stable_formulation_row_id(row)
        text_path = text_path_map.get(key)
        if text_path is None or not text_path.exists():
            pack = {
                "formulation_row_id": row_id,
                "paper_key": key,
                "strict_evidence_units": [],
                "supporting_evidence_units": [],
                "rejected_evidence_units": [
                    {
                        "evidence_id": f"{row_id}__ev_001",
                        "formulation_row_id": row_id,
                        "source_type": "sentence_span",
                        "source_path": str(text_path.resolve()) if text_path is not None else "",
                        "source_locator": {"asset_type": "clean_text", "sentence_index": "", "char_start": "", "char_end": ""},
                        "raw_text_exact": "",
                        "normalized_text": "",
                        "anchor_status": "failed",
                        "context_before": "",
                        "context_after": "",
                        "field_hints": ["unknown"],
                        "dedupe_key": f"{row_id}|missing_text_asset",
                        "is_valid_anchor": False,
                        "is_empty": True,
                        "binding_mode": "unbound",
                        "rejection_reason": "missing_text_asset",
                    }
                ],
            }
            packs.append(pack)
            counts["rows_processed"] += 1
            counts["rejected_units"] += 1
            counts["failed_anchors"] += 1
            audit_rows.append(
                {
                    "formulation_row_id": row_id,
                    "paper_key": key,
                    "evidence_id": f"{row_id}__ev_001",
                    "bucket": "rejected",
                    "source_type": "sentence_span",
                    "binding_mode": "unbound",
                    "anchor_status": "failed",
                    "field_hints": "unknown",
                    "raw_text_exact": "",
                    "rejection_reason": "missing_text_asset",
                    "context_preview": "",
                }
            )
            continue
        pack, pack_audit_rows, pack_counts = process_row(row=row, text_path=text_path, text_cache=text_cache)
        packs.append(pack)
        audit_rows.extend(pack_audit_rows)
        counts.update(pack_counts)
        counts["rows_processed"] += 1

    packs_jsonl = outputs_dir / "stage2_5a_text_evidence_packs_v0.jsonl"
    audit_tsv = outputs_dir / "stage2_5a_text_evidence_units_v0.tsv"
    summary_json = analysis_dir / "stage2_5a_text_evidence_pack_summary_v0.json"
    examples_json = analysis_dir / "stage2_5a_text_evidence_examples_v0.json"
    summary_md = out_dir / "STAGE2_5A_TEXT_EVIDENCE_PACK_V0_VALIDATION_SUMMARY.md"

    with packs_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for pack in packs:
            handle.write(json.dumps(pack, ensure_ascii=False) + "\n")

    write_tsv(
        audit_tsv,
        audit_rows,
        fieldnames=[
            "formulation_row_id",
            "paper_key",
            "evidence_id",
            "bucket",
            "source_type",
            "binding_mode",
            "anchor_status",
            "field_hints",
            "raw_text_exact",
            "rejection_reason",
            "context_preview",
        ],
    )

    examples = summarize_examples(packs)
    summary_payload = {
        "rows_processed": counts["rows_processed"],
        "strict_units": counts["strict_units"],
        "supporting_units": counts["supporting_units"],
        "rejected_units": counts["rejected_units"],
        "failed_anchors": counts["failed_anchors"],
        "out_of_scope_spans": counts["out_of_scope_spans"],
        "boilerplate_rejections": counts["boilerplate_rejections"],
        "unbound_spans": counts["unbound_spans"],
        "selected_paper_keys": subset_keys or sorted({row.get("key", "") for row in rows}),
        "examples": examples,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    examples_json.write_text(json.dumps(examples, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    summary_md.write_text(
        build_validation_summary_md(
            stage2_tsv=stage2_tsv,
            scope_manifest_tsv=scope_manifest_tsv,
            packs_jsonl=packs_jsonl,
            audit_tsv=audit_tsv,
            summary_json=summary_json,
            selected_rows=rows,
            counts=counts,
            examples=examples,
        ),
        encoding="utf-8",
    )

    metadata = build_artifact_metadata(
        source_run_context=run_context,
        source_files={
            "stage2_tsv": str(stage2_tsv),
            "scope_manifest_tsv": str(scope_manifest_tsv),
            "paper_keys_file": str(Path(args.paper_keys_file).resolve()) if normalize_text(args.paper_keys_file) else "",
        },
        generated_by="src/stage2_5_components_shadow/build_text_evidence_packs_v0.py",
        note="Stage2.5A v0 text-only exact-anchored evidence-pack builder",
        extra={
            "outputs": {
                "packs_jsonl": str(packs_jsonl),
                "audit_tsv": str(audit_tsv),
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
            }
        },
    )
    write_artifact_metadata_json(packs_jsonl, metadata)

    print(f"rows_processed\t{counts['rows_processed']}")
    print(f"strict_units\t{counts['strict_units']}")
    print(f"supporting_units\t{counts['supporting_units']}")
    print(f"rejected_units\t{counts['rejected_units']}")
    print(f"failed_anchors\t{counts['failed_anchors']}")
    print(f"out_of_scope_spans\t{counts['out_of_scope_spans']}")
    print(f"packs_jsonl\t{packs_jsonl}")
    print(f"audit_tsv\t{audit_tsv}")
    print(f"summary_md\t{summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
