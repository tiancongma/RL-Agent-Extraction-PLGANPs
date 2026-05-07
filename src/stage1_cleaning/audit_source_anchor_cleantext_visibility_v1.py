#!/usr/bin/env python3
"""Audit governed user-provided source anchors for clean-text visibility.

This module intentionally starts with anchor parsing/inventory support.  The
anchors live in a governed markdown method document and are used as validation
anchors for Stage1/Stage2 visibility diagnostics.  They are not runtime
paper-specific extraction rules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEV15_ANCHOR_KEYS = [
    "INMUTV7L",
    "BB3JUVW7",
    "BXCV5XWB",
    "L3H2RS2H",
    "PA3SPZ28",
    "QLYKLPKT",
    "RHMJWZX8",
    "UFXX9WXE",
    "V99GKZEI",
    "WFDTQ4VX",
    "WIVUCMYG",
    "YGA8VQKU",
    "7ZS858NS",
    "5ZXYABSU",
    "5GIF3D8W",
]

ANCHOR_SECTION_HEADER = "## User-Provided Original Source Excerpts For Field-GT Debugging"


@dataclass(frozen=True)
class AnchorSection:
    paper_key: str
    start_line: int
    end_line: int
    raw_text: str
    has_table_marker: bool
    has_method_marker: bool

    @property
    def raw_line_count(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def raw_sha256(self) -> str:
        return hashlib.sha256((self.raw_text + "\n").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CleanTextVisibilityResult:
    paper_key: str
    clean_text_paths: tuple[str, ...]
    anchor_visibility: str
    matched_fragment_count: int
    missing_fragment_count: int
    first_missing_fragment: str
    audit_note: str
    exact_fragment_match_count: int = 0
    numeric_token_fallback_count: int = 0
    text_source_type: str = ""
    pdf_path: str = ""
    html_path: str = ""
    primary_source_type: str = ""
    secondary_source_available: str = ""
    payload_json_count: int = 0
    normalized_csv_count: int = 0
    grid_cell_count: int = 0
    table_authority_first_failure_class: str = ""
    table_authority_repair_hint: str = ""
    payload_json_path: str = ""
    grid_tsv_path: str = ""
    raw_table_asset_root: str = ""
    raw_table_asset_exists: str = ""
    exact_visibility_proof: str = ""


def normalize_for_visibility(text: str) -> str:
    """Normalize text for visibility-only substring checks.

    This is not row binding and not value authority.  It only reduces common
    source-conversion differences such as unicode minus, micro signs, thin
    spaces, and repeated whitespace.
    """

    replacements = {
        "\u2212": "-",  # unicode minus
        "\u2013": "-",
        "\u2014": "-",
        "\u00b5": "µ",
        "\u03bc": "µ",
        "\u00a0": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\ufeff": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def parse_key2txt(key2txt_path: Path, *, repo_root: Optional[Path] = None) -> Dict[str, list[Path]]:
    """Read key-to-clean-text mapping, preserving multiple source paths per key."""

    repo_root = repo_root or Path.cwd()
    mapping: Dict[str, list[Path]] = {}
    with Path(key2txt_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            key, path_text = parts[0], parts[1]
            path = Path(path_text)
            if not path.is_absolute():
                path = repo_root / path
            mapping.setdefault(key, []).append(path)
    return mapping


def parse_manifest_sources(manifest_path: Path, *, repo_root: Optional[Path] = None) -> Dict[str, dict[str, str]]:
    """Read Stage1 manifest source lineage fields by paper key."""

    repo_root = repo_root or Path.cwd()
    sources: Dict[str, dict[str, str]] = {}
    with Path(manifest_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = row.get("paper_key") or row.get("key")
            if not key:
                continue
            pdf = row.get("pdf", "")
            html = row.get("html", "")
            text_source_type = row.get("text_source_type", "")
            primary = text_source_type
            if not primary:
                text_path = row.get("text_path", "")
                if text_path.endswith(".html.txt"):
                    primary = "html"
                elif text_path.endswith(".pdf.txt"):
                    primary = "pdf"
            secondary_available = "yes" if (pdf and html) else "no"
            sources[key] = {
                "text_source_type": text_source_type,
                "pdf_path": _relpath(Path(pdf), repo_root) if pdf else "",
                "html_path": _relpath(Path(html), repo_root) if html else "",
                "primary_source_type": primary,
                "secondary_source_available": secondary_available,
            }
    return sources


def _anchor_visibility_fragments(anchor: AnchorSection) -> list[str]:
    """Return source-anchor fragments for clean-text visibility checks."""

    fragments: list[str] = []
    for raw_line in anchor.raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("###") or line in {"段落：", "表格：", "材料制备段落：", "表格前面的段落："}:
            continue
        if line.startswith(">"):
            line = line[1:].strip()
        line = line.strip("“”\"")
        normalized = normalize_for_visibility(line)
        if len(normalized) < 12:
            continue
        # Keep lines with formulation-relevant lexical signal or numeric content.
        if not (re.search(r"\d", normalized) or any(token in normalized for token in ("table", "preparation", "prepared", "materials", "formulation"))):
            continue
        fragments.append(line)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for fragment in fragments:
        norm = normalize_for_visibility(fragment)
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(fragment)
    return unique


def _fragment_match_strategy(fragment: str, normalized_fragment: str, normalized_clean_text: str) -> str:
    if normalized_fragment and normalized_fragment in normalized_clean_text:
        return "exact_fragment"
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", normalized_fragment)
    decimal_numbers = [number for number in numbers if "." in number]
    if len(decimal_numbers) >= 2 and sum(number in normalized_clean_text for number in decimal_numbers) >= 2:
        return "numeric_token_fallback"
    if len(numbers) >= 2 and all(number in normalized_clean_text for number in numbers[:6]):
        return "numeric_token_fallback"
    return ""


def _fragment_visible(fragment: str, normalized_fragment: str, normalized_clean_text: str) -> bool:
    return bool(_fragment_match_strategy(fragment, normalized_fragment, normalized_clean_text))


def _compact_visibility_text(text: str) -> str:
    return re.sub(r"[^a-z0-9µ%]+", "", normalize_for_visibility(text))


def _looks_like_payload_geometry_degraded(fragment: str, normalized_fragment: str, normalized_table_text: str) -> bool:
    """Return True when fragment evidence appears present only as disjoint cells.

    This is a diagnostic heuristic only.  It does not prove row binding or value
    authority; it identifies likely S2-2a row/header geometry loss for repair
    triage.
    """

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", normalized_fragment)
    lexical_tokens = [
        token
        for token in re.findall(r"[a-zµ]{4,}", normalized_fragment)
        if token not in {"table", "formulation", "prepared", "preparation", "materials", "method"}
    ]
    number_hits = sum(number in normalized_table_text for number in numbers)
    lexical_hits = sum(token in normalized_table_text for token in lexical_tokens[:8])
    return bool(numbers) and number_hits >= min(len(numbers), 2) and lexical_hits >= 1


def classify_table_authority_first_failure(
    *,
    anchor: AnchorSection,
    fragments: list[str],
    normalized_table_text: str,
    exact_matches: int,
    numeric_token_fallbacks: int,
    payload_json_count: int,
    normalized_csv_count: int,
    grid_cell_count: int,
    raw_table_asset_exists: bool,
) -> tuple[str, str, str]:
    """Classify table-authority visibility failures into actionable buckets.

    Numeric-token fallback is intentionally reported as signal only and never as
    exact visibility proof.
    """

    has_payload_or_grid = bool(payload_json_count or normalized_csv_count or grid_cell_count)
    if not fragments:
        return "no_checkable_fragments", "no_runtime_repair_from_empty_anchor_visibility_surface", "no"
    if exact_matches == len(fragments):
        return "exact_visible", "no_table_authority_visibility_repair_indicated", "yes"
    if not anchor.has_table_marker:
        return "source_excerpt_method_prose_not_expected_in_table_payload", "route_to_clean_text_or_prompt_semantic_adequacy_diagnostic", "no"
    if not raw_table_asset_exists and not has_payload_or_grid:
        return "raw_table_asset_missing", "repair_stage1_table_extraction_or_manifest_table_asset_binding", "no"
    if raw_table_asset_exists and not has_payload_or_grid:
        return "raw_table_asset_exists_but_s2_2a_payload_missing", "repair_s2_2a_table_payload_generation_or_alias_binding", "no"
    compact_table = _compact_visibility_text(normalized_table_text)
    for fragment in fragments:
        norm = normalize_for_visibility(fragment)
        if norm and norm not in normalized_table_text and _compact_visibility_text(norm) in compact_table:
            return "payload_exists_but_text_normalization_mismatch", "repair_visibility_normalization_or_payload_text_projection", "no"
    if any(_looks_like_payload_geometry_degraded(fragment, normalize_for_visibility(fragment), normalized_table_text) for fragment in fragments):
        return "payload_exists_but_row_header_geometry_degraded", "repair_s2_2a_header_row_geometry_preservation", "no"
    if numeric_token_fallbacks:
        return "table_asset_exists_but_wrong_table_selected_or_overcompressed", "inspect_table_alias_selection_and_summary_compaction", "no"
    return "table_asset_exists_but_wrong_table_selected", "repair_table_label_caption_alias_rebinding_or_selector_retention", "no"


def audit_anchor_clean_text_visibility(
    anchor: AnchorSection,
    clean_text_paths: Iterable[Path],
) -> CleanTextVisibilityResult:
    """Audit whether anchor fragments are visible in one or more clean-text files."""

    paths = [Path(path) for path in clean_text_paths]
    combined_parts: list[str] = []
    existing_paths: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        existing_paths.append(str(path))
        combined_parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    combined = normalize_for_visibility("\n".join(combined_parts))
    fragments = _anchor_visibility_fragments(anchor)
    matched = 0
    exact_matches = 0
    numeric_fallbacks = 0
    first_missing = ""
    for fragment in fragments:
        norm = normalize_for_visibility(fragment)
        strategy = _fragment_match_strategy(fragment, norm, combined)
        if strategy:
            matched += 1
            if strategy == "exact_fragment":
                exact_matches += 1
            elif strategy == "numeric_token_fallback":
                numeric_fallbacks += 1
        elif not first_missing:
            first_missing = fragment[:240]
    missing = max(len(fragments) - matched, 0)
    if not fragments:
        visibility = "no_checkable_fragments"
    elif matched == len(fragments):
        visibility = "full"
    elif matched > 0:
        visibility = "partial"
    else:
        visibility = "absent"
    return CleanTextVisibilityResult(
        paper_key=anchor.paper_key,
        clean_text_paths=tuple(existing_paths or [str(path) for path in paths]),
        anchor_visibility=visibility,
        matched_fragment_count=matched,
        missing_fragment_count=missing,
        first_missing_fragment=first_missing,
        audit_note="visibility_only_not_row_binding_not_stage5_materialization",
        exact_fragment_match_count=exact_matches,
        numeric_token_fallback_count=numeric_fallbacks,
    )

def audit_anchor_table_authority_visibility(
    anchor: AnchorSection,
    *,
    payload_root: Path,
    grid_tsv_path: Path,
    repo_root: Optional[Path] = None,
    raw_table_asset_root: Optional[Path] = None,
) -> CleanTextVisibilityResult:
    """Audit anchor visibility in execution-grade normalized table payload/grid artifacts."""

    repo_root = repo_root or Path.cwd()
    combined_parts: list[str] = []
    payload_json_count = 0
    normalized_csv_count = 0
    paper_payload_json = Path(payload_root) / anchor.paper_key / "normalized_table_payloads_v1.json"
    raw_root = Path(raw_table_asset_root) / anchor.paper_key if raw_table_asset_root else Path(payload_root) / anchor.paper_key
    raw_table_asset_exists = raw_root.exists() and any(raw_root.iterdir())
    if paper_payload_json.exists():
        payload_json_count = 1
        payload_text = paper_payload_json.read_text(encoding="utf-8", errors="ignore")
        combined_parts.append(payload_text)
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            payload = {}
        for item in payload.get("normalized_table_payloads", []) if isinstance(payload, dict) else []:
            csv_path_text = item.get("normalized_csv_path") or item.get("source_csv_path") or ""
            if not csv_path_text:
                continue
            csv_path = Path(csv_path_text)
            if not csv_path.is_absolute():
                csv_path = repo_root / csv_path
            if not csv_path.exists():
                alt = paper_payload_json.parent / Path(csv_path_text).name
                csv_path = alt if alt.exists() else csv_path
            if csv_path.exists():
                normalized_csv_count += 1
                combined_parts.append(csv_path.read_text(encoding="utf-8", errors="ignore"))

    grid_cell_count = 0
    if Path(grid_tsv_path).exists():
        with Path(grid_tsv_path).open(encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row.get("paper_key") != anchor.paper_key:
                    continue
                grid_cell_count += 1
                combined_parts.append(" ".join(str(row.get(col, "")) for col in ("raw_header_text", "raw_cell_value", "row_label_candidate", "column_label_candidate", "source_caption_or_title")))

    combined = normalize_for_visibility("\n".join(combined_parts))
    fragments = _anchor_visibility_fragments(anchor)
    matched = 0
    exact_matches = 0
    numeric_fallbacks = 0
    first_missing = ""
    for fragment in fragments:
        norm = normalize_for_visibility(fragment)
        strategy = _fragment_match_strategy(fragment, norm, combined)
        if strategy == "exact_fragment":
            matched += 1
            exact_matches += 1
        elif strategy == "numeric_token_fallback":
            numeric_fallbacks += 1
            if not first_missing:
                first_missing = fragment[:240]
        elif not first_missing:
            first_missing = fragment[:240]
    missing = max(len(fragments) - matched, 0)
    if not fragments:
        visibility = "no_checkable_fragments"
    elif matched == len(fragments):
        visibility = "full"
    elif matched > 0:
        visibility = "partial"
    else:
        visibility = "absent"
    first_failure_class, repair_hint, exact_visibility_proof = classify_table_authority_first_failure(
        anchor=anchor,
        fragments=fragments,
        normalized_table_text=combined,
        exact_matches=exact_matches,
        numeric_token_fallbacks=numeric_fallbacks,
        payload_json_count=payload_json_count,
        normalized_csv_count=normalized_csv_count,
        grid_cell_count=grid_cell_count,
        raw_table_asset_exists=raw_table_asset_exists,
    )
    return CleanTextVisibilityResult(
        paper_key=anchor.paper_key,
        clean_text_paths=tuple(),
        anchor_visibility=visibility,
        matched_fragment_count=matched,
        missing_fragment_count=missing,
        first_missing_fragment=first_missing,
        audit_note="table_authority_visibility_only_not_row_binding_not_value_authority_not_stage5_materialization_numeric_fallback_signal_only",
        exact_fragment_match_count=exact_matches,
        numeric_token_fallback_count=numeric_fallbacks,
        payload_json_count=payload_json_count,
        normalized_csv_count=normalized_csv_count,
        grid_cell_count=grid_cell_count,
        table_authority_first_failure_class=first_failure_class,
        table_authority_repair_hint=repair_hint,
        payload_json_path=_relpath(paper_payload_json, repo_root),
        grid_tsv_path=_relpath(Path(grid_tsv_path), repo_root),
        raw_table_asset_root=_relpath(raw_root, repo_root),
        raw_table_asset_exists="yes" if raw_table_asset_exists else "no",
        exact_visibility_proof=exact_visibility_proof,
    )


def _has_method_marker(raw_text: str) -> bool:
    lower = raw_text.lower()
    return any(
        token in lower
        for token in (
            "method",
            "methods",
            "preparation",
            "prepared",
            "fabrication",
            "materials",
            "synthesis",
        )
    )


def _has_table_marker(raw_text: str) -> bool:
    lower = raw_text.lower()
    return "表格" in raw_text or "table" in lower or "\t" in raw_text


def parse_user_source_anchor_sections(
    protocol_path: Path,
    *,
    expected_keys: Optional[Iterable[str]] = None,
) -> List[AnchorSection]:
    """Parse governed DEV15 source anchors from ``protocol_path``.

    Parsing is deliberately conservative: after the governed anchor section
    begins, only ``### <8-char-paper-key>`` headers from ``expected_keys`` are
    accepted as anchor headers.  The first non-paper-key ``###`` heading after
    anchors starts non-anchor diagnostic/method notes and terminates parsing.
    This prevents later repair notes from being swallowed into the last source
    anchor.
    """

    expected_order = list(expected_keys or DEV15_ANCHOR_KEYS)
    expected_set = set(expected_order)
    lines = Path(protocol_path).read_text(encoding="utf-8", errors="ignore").splitlines()

    section_start: Optional[int] = None
    for idx, line in enumerate(lines, start=1):
        if line.strip() == ANCHOR_SECTION_HEADER:
            section_start = idx
            break
    if section_start is None:
        raise ValueError(f"anchor section not found in {protocol_path}")

    headers: list[tuple[int, str]] = []
    stop_line: Optional[int] = None
    for idx in range(section_start + 1, len(lines) + 1):
        line = lines[idx - 1]
        if idx > section_start and line.startswith("## "):
            stop_line = idx
            break
        match = re.match(r"^###\s+(.+?)\s*$", line)
        if not match:
            continue
        title = match.group(1).strip()
        if re.fullmatch(r"[A-Z0-9]{8}", title):
            if title not in expected_set:
                raise ValueError(
                    f"unexpected paper-key-like anchor header {title!r} at line {idx}; "
                    "expected_keys may be incomplete or the governed anchor list changed"
                )
            headers.append((idx, title))
            continue
        stop_line = idx
        break

    if [key for _, key in headers] != expected_order:
        raise ValueError(
            "DEV15 anchor keys not found in expected order: "
            f"found={[key for _, key in headers]} expected={expected_order}"
        )

    anchors: list[AnchorSection] = []
    for pos, (start_line, paper_key) in enumerate(headers):
        if pos + 1 < len(headers):
            end_line = headers[pos + 1][0] - 1
        elif stop_line is not None:
            end_line = stop_line - 1
        else:
            end_line = len(lines)
        raw_text = "\n".join(lines[start_line - 1 : end_line])
        anchors.append(
            AnchorSection(
                paper_key=paper_key,
                start_line=start_line,
                end_line=end_line,
                raw_text=raw_text,
                has_table_marker=_has_table_marker(raw_text),
                has_method_marker=_has_method_marker(raw_text),
            )
        )
    return anchors


def write_anchor_inventory(
    anchors: list[AnchorSection],
    *,
    protocol_path: Path,
    out_dir: Path,
    repo_root: Optional[Path] = None,
) -> None:
    """Write diagnostic-only anchor inventory artifacts."""

    repo_root = repo_root or Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_anchor_snippets"
    raw_dir.mkdir(parents=True, exist_ok=True)

    source_rel = _relpath(protocol_path, repo_root)
    inv_path = out_dir / "source_anchor_inventory_v1.tsv"
    with inv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "paper_key",
            "anchor_start_line",
            "anchor_end_line",
            "has_method_paragraph",
            "has_table",
            "anchor_source_file",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for anchor in anchors:
            writer.writerow(
                {
                    "paper_key": anchor.paper_key,
                    "anchor_start_line": anchor.start_line,
                    "anchor_end_line": anchor.end_line,
                    "has_method_paragraph": "yes" if anchor.has_method_marker else "no",
                    "has_table": "yes" if anchor.has_table_marker else "no",
                    "anchor_source_file": source_rel,
                }
            )

    manifest_path = out_dir / "source_anchor_raw_snippet_manifest_v1.tsv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["paper_key", "raw_snippet_file", "raw_sha256", "raw_line_count"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for anchor in anchors:
            raw_path = raw_dir / f"{anchor.paper_key}_raw_excerpt.md"
            raw_path.write_text(anchor.raw_text + "\n", encoding="utf-8")
            writer.writerow(
                {
                    "paper_key": anchor.paper_key,
                    "raw_snippet_file": _relpath(raw_path, repo_root),
                    "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    "raw_line_count": anchor.raw_line_count,
                }
            )

    with (out_dir / "source_anchor_inventory_summary_v1.tsv").open("w", encoding="utf-8") as handle:
        handle.write("metric\tvalue\n")
        handle.write(f"anchor_count\t{len(anchors)}\n")
        handle.write(
            "anchors_with_method_paragraph\t"
            f"{sum(anchor.has_method_marker for anchor in anchors)}\n"
        )
        handle.write(f"anchors_with_table\t{sum(anchor.has_table_marker for anchor in anchors)}\n")
        no_table = [anchor.paper_key for anchor in anchors if not anchor.has_table_marker]
        handle.write(f"anchors_without_table_marker\t{len(no_table)}\n")
        handle.write(f"anchors_without_table_marker_keys\t{','.join(no_table)}\n")

    metadata = {
        "generated_by": "audit_source_anchor_cleantext_visibility_v1.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "diagnostic_only": True,
        "benchmark_valid": "no",
        "anchor_source_file": source_rel,
        "source_excerpt_section_start_line": _find_anchor_section_start(protocol_path),
        "source_excerpt_last_anchor_end_line": anchors[-1].end_line if anchors else None,
        "excluded_non_anchor_note_start_line": (anchors[-1].end_line + 1) if anchors else None,
        "expected_dev15_anchor_count": len(DEV15_ANCHOR_KEYS),
        "actual_anchor_count": len(anchors),
        "inventory_schema": "paper_key,anchor_start_line,anchor_end_line,has_method_paragraph,has_table,anchor_source_file",
        "raw_snippet_manifest": "source_anchor_raw_snippet_manifest_v1.tsv",
        "has_table_caveat": (
            "has_table=no means no explicit table marker/table block in governed anchor; "
            "prose numeric evidence may still exist and must not be skipped"
        ),
    }
    (out_dir / "source_anchor_inventory_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _find_anchor_section_start(protocol_path: Path) -> int:
    for idx, line in enumerate(Path(protocol_path).read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if line.strip() == ANCHOR_SECTION_HEADER:
            return idx
    raise ValueError(f"anchor section not found in {protocol_path}")


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def write_clean_text_visibility_audit(
    anchors: list[AnchorSection],
    *,
    key2txt_path: Path,
    out_dir: Path,
    repo_root: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> None:
    """Write diagnostic clean-text visibility audit for governed anchors."""

    repo_root = repo_root or Path.cwd()
    mapping = parse_key2txt(key2txt_path, repo_root=repo_root)
    lineage = parse_manifest_sources(manifest_path, repo_root=repo_root) if manifest_path else {}
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[CleanTextVisibilityResult] = []
    for anchor in anchors:
        result = audit_anchor_clean_text_visibility(anchor, mapping.get(anchor.paper_key, []))
        if anchor.paper_key in lineage:
            result = replace(result, **lineage[anchor.paper_key])
        rows.append(result)

    visibility_path = out_dir / "source_anchor_cleantext_visibility_v1.tsv"
    with visibility_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "paper_key",
            "clean_text_paths",
            "anchor_visibility",
            "matched_fragment_count",
            "exact_fragment_match_count",
            "numeric_token_fallback_count",
            "missing_fragment_count",
            "first_missing_fragment",
            "text_source_type",
            "pdf_path",
            "html_path",
            "primary_source_type",
            "secondary_source_available",
            "audit_note",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for result in rows:
            writer.writerow(
                {
                    "paper_key": result.paper_key,
                    "clean_text_paths": ";".join(_relpath(Path(path), repo_root) for path in result.clean_text_paths),
                    "anchor_visibility": result.anchor_visibility,
                    "matched_fragment_count": result.matched_fragment_count,
                    "exact_fragment_match_count": result.exact_fragment_match_count,
                    "numeric_token_fallback_count": result.numeric_token_fallback_count,
                    "missing_fragment_count": result.missing_fragment_count,
                    "first_missing_fragment": result.first_missing_fragment.replace("\t", " ").replace("\n", " "),
                    "text_source_type": result.text_source_type,
                    "pdf_path": result.pdf_path,
                    "html_path": result.html_path,
                    "primary_source_type": result.primary_source_type,
                    "secondary_source_available": result.secondary_source_available,
                    "audit_note": result.audit_note,
                }
            )

    counts: Dict[str, int] = {}
    for result in rows:
        counts[result.anchor_visibility] = counts.get(result.anchor_visibility, 0) + 1
    with (out_dir / "source_anchor_cleantext_visibility_summary_v1.tsv").open("w", encoding="utf-8") as handle:
        handle.write("metric\tvalue\n")
        handle.write(f"anchor_count\t{len(rows)}\n")
        for status in sorted(counts):
            handle.write(f"anchor_visibility.{status}\t{counts[status]}\n")
        handle.write(f"total_matched_fragments\t{sum(r.matched_fragment_count for r in rows)}\n")
        handle.write(f"total_exact_fragment_matches\t{sum(r.exact_fragment_match_count for r in rows)}\n")
        handle.write(f"total_numeric_token_fallback_matches\t{sum(r.numeric_token_fallback_count for r in rows)}\n")
        handle.write(f"total_missing_fragments\t{sum(r.missing_fragment_count for r in rows)}\n")
        absent_keys = [r.paper_key for r in rows if r.anchor_visibility == "absent"]
        handle.write(f"absent_anchor_keys\t{','.join(absent_keys)}\n")

    generated_at = datetime.now().isoformat(timespec="seconds")
    metadata = {
        "generated_by": "audit_source_anchor_cleantext_visibility_v1.py",
        "generated_at": generated_at,
        "diagnostic_only": True,
        "benchmark_valid": "no",
        "key2txt_path": _relpath(key2txt_path, repo_root),
        "manifest_path": _relpath(manifest_path, repo_root) if manifest_path else "",
        "visibility_semantics": "visibility-only exact-fragment plus explicitly counted numeric-token fallback audit; not row binding, not value authority, not Stage5 materialization",
        "outputs": [
            "source_anchor_cleantext_visibility_v1.tsv",
            "source_anchor_cleantext_visibility_summary_v1.tsv",
            "source_anchor_cleantext_visibility_metadata.json",
        ],
    }
    (out_dir / "source_anchor_cleantext_visibility_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "RUN_CONTEXT.md").write_text(
        "# source_anchor_cleantext_visibility_diagnostic\n\n"
        "Diagnostic-only clean-text visibility audit for governed user-provided source anchors.\n\n"
        "- benchmark_valid: no\n"
        "- diagnostic_only: yes\n"
        f"- generated_at: {generated_at}\n"
        f"- key2txt_path: {_relpath(key2txt_path, repo_root)}\n"
        f"- manifest_path: {_relpath(manifest_path, repo_root) if manifest_path else ''}\n"
        "- visibility_semantics: visibility-only exact-fragment plus counted numeric-token fallback; not row binding, not value authority, not Stage5 materialization\n"
        f"- anchor_count: {len(rows)}\n"
        f"- full: {counts.get('full', 0)}\n"
        f"- partial: {counts.get('partial', 0)}\n"
        f"- absent: {counts.get('absent', 0)}\n\n"
        "Outputs:\n"
        "- source_anchor_cleantext_visibility_v1.tsv\n"
        "- source_anchor_cleantext_visibility_summary_v1.tsv\n"
        "- source_anchor_cleantext_visibility_metadata.json\n\n"
        "This artifact checks whether governed anchor fragments are visible in active clean text. It does not compare GT, infer row binding, or repair runtime behavior.\n",
        encoding="utf-8",
    )


def write_table_authority_visibility_audit(
    anchors: list[AnchorSection],
    *,
    payload_root: Path,
    grid_tsv_path: Path,
    out_dir: Path,
    repo_root: Optional[Path] = None,
    raw_table_asset_root: Optional[Path] = None,
) -> None:
    """Write diagnostic table-authority visibility audit for governed anchors."""

    repo_root = repo_root or Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        audit_anchor_table_authority_visibility(
            anchor,
            payload_root=payload_root,
            grid_tsv_path=grid_tsv_path,
            repo_root=repo_root,
            raw_table_asset_root=raw_table_asset_root,
        )
        for anchor in anchors
    ]

    fieldnames = [
        "paper_key",
        "anchor_visibility",
        "table_authority_first_failure_class",
        "table_authority_repair_hint",
        "exact_visibility_proof",
        "matched_fragment_count",
        "exact_fragment_match_count",
        "numeric_token_fallback_count",
        "missing_fragment_count",
        "first_missing_fragment",
        "payload_json_count",
        "normalized_csv_count",
        "grid_cell_count",
        "payload_json_path",
        "grid_tsv_path",
        "raw_table_asset_root",
        "raw_table_asset_exists",
        "audit_note",
    ]
    with (out_dir / "source_anchor_table_authority_visibility_v1.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for result in rows:
            writer.writerow(
                {
                    "paper_key": result.paper_key,
                    "anchor_visibility": result.anchor_visibility,
                    "table_authority_first_failure_class": result.table_authority_first_failure_class,
                    "table_authority_repair_hint": result.table_authority_repair_hint,
                    "exact_visibility_proof": result.exact_visibility_proof,
                    "matched_fragment_count": result.matched_fragment_count,
                    "exact_fragment_match_count": result.exact_fragment_match_count,
                    "numeric_token_fallback_count": result.numeric_token_fallback_count,
                    "missing_fragment_count": result.missing_fragment_count,
                    "first_missing_fragment": result.first_missing_fragment.replace("\t", " ").replace("\n", " "),
                    "payload_json_count": result.payload_json_count,
                    "normalized_csv_count": result.normalized_csv_count,
                    "grid_cell_count": result.grid_cell_count,
                    "payload_json_path": result.payload_json_path,
                    "grid_tsv_path": result.grid_tsv_path,
                    "raw_table_asset_root": result.raw_table_asset_root,
                    "raw_table_asset_exists": result.raw_table_asset_exists,
                    "audit_note": result.audit_note,
                }
            )

    counts: Dict[str, int] = {}
    failure_counts: Dict[str, int] = {}
    for result in rows:
        counts[result.anchor_visibility] = counts.get(result.anchor_visibility, 0) + 1
        failure_counts[result.table_authority_first_failure_class] = failure_counts.get(result.table_authority_first_failure_class, 0) + 1
    with (out_dir / "source_anchor_table_authority_visibility_summary_v1.tsv").open("w", encoding="utf-8") as handle:
        handle.write("metric\tvalue\n")
        handle.write(f"anchor_count\t{len(rows)}\n")
        for status in sorted(counts):
            handle.write(f"anchor_visibility.{status}\t{counts[status]}\n")
        for failure_class in sorted(failure_counts):
            handle.write(f"table_authority_first_failure_class.{failure_class}\t{failure_counts[failure_class]}\n")
        handle.write(f"total_matched_fragments\t{sum(r.matched_fragment_count for r in rows)}\n")
        handle.write(f"total_exact_fragment_matches\t{sum(r.exact_fragment_match_count for r in rows)}\n")
        handle.write(f"total_numeric_token_fallback_matches\t{sum(r.numeric_token_fallback_count for r in rows)}\n")
        handle.write(f"total_missing_fragments\t{sum(r.missing_fragment_count for r in rows)}\n")
        handle.write(f"anchors_with_payload_json\t{sum(1 for r in rows if r.payload_json_count)}\n")
        handle.write(f"anchors_with_grid_cells\t{sum(1 for r in rows if r.grid_cell_count)}\n")
        absent_keys = [r.paper_key for r in rows if r.anchor_visibility == "absent"]
        handle.write(f"absent_anchor_keys\t{','.join(absent_keys)}\n")

    generated_at = datetime.now().isoformat(timespec="seconds")
    metadata = {
        "generated_by": "audit_source_anchor_cleantext_visibility_v1.py",
        "generated_at": generated_at,
        "diagnostic_only": True,
        "benchmark_valid": "no",
        "payload_root": _relpath(payload_root, repo_root),
        "grid_tsv_path": _relpath(grid_tsv_path, repo_root),
        "raw_table_asset_root": _relpath(raw_table_asset_root, repo_root) if raw_table_asset_root else _relpath(payload_root, repo_root),
        "visibility_semantics": "table-authority visibility-only exact-fragment audit; numeric-token fallback is signal-only, not visibility proof; not row binding, not value authority, not Stage5 materialization",
        "outputs": [
            "source_anchor_table_authority_visibility_v1.tsv",
            "source_anchor_table_authority_visibility_summary_v1.tsv",
            "source_anchor_table_authority_visibility_metadata.json",
        ],
    }
    (out_dir / "source_anchor_table_authority_visibility_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "RUN_CONTEXT.md").write_text(
        "# source_anchor_table_authority_visibility_diagnostic\n\n"
        "Diagnostic-only table-authority visibility audit for governed user-provided source anchors.\n\n"
        "- benchmark_valid: no\n"
        "- diagnostic_only: yes\n"
        f"- generated_at: {generated_at}\n"
        f"- payload_root: {_relpath(payload_root, repo_root)}\n"
        f"- grid_tsv_path: {_relpath(grid_tsv_path, repo_root)}\n"
        f"- raw_table_asset_root: {_relpath(raw_table_asset_root, repo_root) if raw_table_asset_root else _relpath(payload_root, repo_root)}\n"
        "- visibility_semantics: table-authority visibility-only exact-fragment; numeric-token fallback is signal-only, not visibility proof; not row binding, not value authority, not Stage5 materialization\n"
        f"- anchor_count: {len(rows)}\n"
        f"- full: {counts.get('full', 0)}\n"
        f"- partial: {counts.get('partial', 0)}\n"
        f"- absent: {counts.get('absent', 0)}\n\n"
        "Outputs:\n"
        "- source_anchor_table_authority_visibility_v1.tsv\n"
        "- source_anchor_table_authority_visibility_summary_v1.tsv\n"
        "- source_anchor_table_authority_visibility_metadata.json\n\n"
        "This artifact checks whether governed anchor fragments are visible in execution-grade table payload/grid artifacts. It does not compare GT, infer row binding, or repair runtime behavior.\n",
        encoding="utf-8",
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-md", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--key2txt", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--payload-root", type=Path)
    parser.add_argument("--grid-tsv", type=Path)
    parser.add_argument("--raw-table-asset-root", type=Path)
    parser.add_argument("--write-inventory-only", action="store_true")
    args = parser.parse_args(argv)

    anchors = parse_user_source_anchor_sections(args.protocol_md)
    write_anchor_inventory(anchors, protocol_path=args.protocol_md, out_dir=args.out_dir, repo_root=args.repo_root)
    if args.key2txt and not args.write_inventory_only:
        write_clean_text_visibility_audit(
            anchors,
            key2txt_path=args.key2txt,
            out_dir=args.out_dir,
            repo_root=args.repo_root,
            manifest_path=args.manifest,
        )
    if args.payload_root and args.grid_tsv and not args.write_inventory_only:
        write_table_authority_visibility_audit(
            anchors,
            payload_root=args.payload_root,
            grid_tsv_path=args.grid_tsv,
            out_dir=args.out_dir,
            repo_root=args.repo_root,
            raw_table_asset_root=args.raw_table_asset_root,
        )
    print(f"wrote {len(anchors)} source anchors to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
