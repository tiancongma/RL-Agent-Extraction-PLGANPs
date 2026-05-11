#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FIELDS = [
    "paper_key",
    "field_family",
    "surface_form",
    "canonical_form",
    "alias_type",
    "scope",
    "paper_key_scope",
    "normalization_rule",
    "status",
    "promotion_status",
    "source_stage",
    "confidence",
    "evidence_quote",
    "evidence_location",
    "collision_notes",
]

DRUG_CONTEXT_RE = re.compile(
    r"\b(?:drug|loaded|loading|encapsulat(?:ed|ion)|payload|active|therapeutic|model\s+drug)\b",
    re.IGNORECASE,
)
MATERIAL_CONTEXT_RE = re.compile(r"\b(?:surfactant|stabilizer|emulsifier|polymer|solvent)\b", re.IGNORECASE)
ABBR_RE = re.compile(
    r"(?P<full>[A-Z][A-Za-z0-9α-ωΑ-Ωβγδκ\-/]+(?:\s+[A-Za-z0-9α-ωΑ-Ωβγδκ\-/]+){1,6})\s*\((?P<abbr>[A-Z][A-Z0-9\-]{1,12})\)"
)
REVERSE_ABBR_RE = re.compile(
    r"(?P<abbr>[A-Z][A-Z0-9\-]{1,12})\s*\((?P<full>[A-Z][A-Za-z0-9α-ωΑ-Ωβγδκ\-/]+(?:\s+[A-Za-z0-9α-ωΑ-Ωβγδκ\-/]+){1,6})\)"
)
BLOCKED_ABBR_RE = re.compile(r"^(?:F\d+|P\d+|NP\d+|BATCH\d*|T\d+|R\d+)$", re.IGNORECASE)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value).lower())


def infer_field_family(full_term: str, context: str) -> str:
    window = normalize_text(context)
    full = normalize_text(full_term).lower()
    if DRUG_CONTEXT_RE.search(window):
        return "drug_name"
    if MATERIAL_CONTEXT_RE.search(window):
        if "surfactant" in window.lower() or "emulsifier" in window.lower():
            return "surfactant_name"
        if "stabilizer" in window.lower():
            return "stabilizer_name"
        if "polymer" in window.lower() or "plga" in full:
            return "polymer_name"
        if "solvent" in window.lower():
            return "solvent_name"
    # Conservative PLGA-domain default for title/abstract definitions: keep as
    # paper-local drug candidate rather than promote globally.
    return "drug_name"


def _quote_window(text: str, start: int, end: int, *, radius: int = 80) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return normalize_text(text[left:right])


def _clean_full_term(value: str) -> str:
    text = normalize_text(value)
    text = re.sub(r"^(?:the\s+)?(?:model\s+)?(?:drug|payload|active|therapeutic|compound)\s+", "", text, flags=re.IGNORECASE)
    return normalize_text(text)


def _candidate_from_match(paper_key: str, text: str, match: re.Match[str]) -> dict[str, str] | None:
    abbr = normalize_text(match.group("abbr"))
    full = _clean_full_term(match.group("full"))
    if not abbr or not full or BLOCKED_ABBR_RE.fullmatch(abbr):
        return None
    if len(abbr) > 12 or len(full.split()) > 8:
        return None
    quote = _quote_window(text, match.start(), match.end())
    field_family = infer_field_family(full, quote)
    confidence = "0.80" if field_family == "drug_name" and DRUG_CONTEXT_RE.search(quote) else "0.65"
    return {
        "paper_key": normalize_text(paper_key),
        "field_family": field_family,
        "surface_form": abbr,
        "canonical_form": full,
        "alias_type": "abbreviation",
        "scope": "paper_local",
        "paper_key_scope": normalize_text(paper_key),
        "normalization_rule": "exact",
        "status": "candidate",
        "promotion_status": "local_only",
        "source_stage": "stage2_abbreviation_registry_v1",
        "confidence": confidence,
        "evidence_quote": quote,
        "evidence_location": f"char:{match.start()}-{match.end()}",
        "collision_notes": "",
    }


def extract_abbreviation_candidates_from_text(*, paper_key: str, text: str) -> list[dict[str, str]]:
    text = str(text or "")
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for pattern in (ABBR_RE, REVERSE_ABBR_RE):
        for match in pattern.finditer(text):
            candidate = _candidate_from_match(paper_key, text, match)
            if not candidate:
                continue
            key = (
                candidate["field_family"],
                _canonical_key(candidate["surface_form"]),
                _canonical_key(candidate["canonical_form"]),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates


def read_text_inputs(paths: list[Path]) -> list[tuple[str, str, str]]:
    inputs: list[tuple[str, str, str]] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.glob("*.txt")):
                inputs.append((child.stem, child.read_text(encoding="utf-8", errors="replace"), str(child)))
        elif path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    paper_key = normalize_text(obj.get("paper_key") or obj.get("key") or obj.get("id"))
                    text = normalize_text(obj.get("text") or obj.get("clean_text") or obj.get("content") or obj.get("evidence_text"))
                    if paper_key and text:
                        inputs.append((paper_key, text, f"{path}:{line_no}"))
        else:
            inputs.append((path.stem, path.read_text(encoding="utf-8", errors="replace"), str(path)))
    return inputs


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build run-scoped paper-local abbreviation dictionary candidates.")
    parser.add_argument("--input", action="append", default=[], help="Text file, text directory, or JSONL containing paper_key/text fields. Repeatable.")
    parser.add_argument("--output", required=True, help="Output paper-local abbreviation registry TSV.")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for paper_key, text, location in read_text_inputs([Path(item) for item in args.input]):
        for row in extract_abbreviation_candidates_from_text(paper_key=paper_key, text=text):
            row["evidence_location"] = f"{location}:{row['evidence_location']}"
            rows.append(row)
    rows.sort(key=lambda row: (row["paper_key"], row["field_family"], row["surface_form"], row["canonical_form"]))
    write_tsv(Path(args.output), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
