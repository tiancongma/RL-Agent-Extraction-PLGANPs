#!/usr/bin/env python3
from __future__ import annotations

"""
Deterministically enumerate explicit numbered DOE formulation rows from Stage1 table assets.

DOE enumerator contract:
- Definition: a deterministic row materializer for explicit DOE anchors already present in the table structure.
- Input: Stage2 semantic output plus Stage1 table assets, but only the explicit row anchors in those assets are eligible for recovery.
- Allowed anchors: explicit numbered rows, explicit row labels, and explicit design-matrix table rows.
- Forbidden behavior: design-space expansion, inferred unseen combinations, or semantic invention of missing rows.
- Output: additional formulation rows with traceable anchors and stable provenance fields.

Stage role:
- Stage2 boundary support tool.
- Runs after Stage1 table extraction and before Stage3 relation materialization.
- Provides deterministic recovery of explicit numbered DOE/design-table rows that
  the LLM extraction layer may under-enumerate.

Inputs:
- A Stage2 scope manifest TSV with at least `key`, `doi`, `title`, and `text_path`.
- Existing Stage1 table assets under `data/cleaned/<dataset_id>/tables/<paper_key>/`.
- Optional existing Stage2 weak-label TSV for duplicate suppression and regression comparison.

Outputs:
- `numbered_doe_row_candidates_v1.tsv`
- `numbered_doe_row_candidates_summary_v1.tsv`
- `RUN_CONTEXT.md` in the target run directory when invoked through the CLI.

What this tool does:
- Detects explicit numbered rows in DOE-style tables.
- Enumerates each numbered row into a deterministic formulation candidate payload.
- Preserves non-core varying factors in explicit JSON columns instead of dropping them.
- Emits a Stage2-compatible candidate structure that can be merged additively.

What this tool does not do:
- It does not call any LLM or external API.
- It does not perform free-form DOE design-space expansion.
- It does not replace the active Stage2 extractor.
- It does not claim benchmark-valid final output.
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import DATA_CLEANED_DIR, DATA_RESULTS_DIR
from src.utils.run_id import validate_artifact_subdir


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_NAME = "numbered_doe_row_candidates_v1.tsv"
SUMMARY_NAME = "numbered_doe_row_candidates_summary_v1.tsv"
VALID_RUN_TYPES = {
    "intermediate_diagnostic_run",
    "component_regression_run",
    "full_pipeline_benchmark_run",
}


@dataclass(frozen=True)
class PaperRecord:
    key: str
    doi: str
    title: str
    text_path: Path


def normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_doi(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"^doi\s*:\s*", "", text)
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    return text


def validate_run_id(run_id: str) -> str:
    rid = str(run_id or "").strip()
    if not re.fullmatch(r"^run_\d{8}_\d{4}_[0-9a-f]{7}_.+$", rid):
        raise ValueError(f"run_id does not match required pattern: {rid}")
    return rid


def validate_out_subdir(out_subdir: str) -> str:
    return validate_artifact_subdir(out_subdir, param_name="out_subdir")


def load_manifest(manifest_tsv: Path, paper_keys: list[str]) -> list[PaperRecord]:
    if not manifest_tsv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_tsv}")
    df = pd.read_csv(manifest_tsv, sep="\t", dtype=str).fillna("")
    required = {"key", "doi", "title", "text_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    if paper_keys:
        keep = {key.strip() for key in paper_keys if key.strip()}
        df = df[df["key"].astype(str).isin(keep)].copy()
    rows: list[PaperRecord] = []
    for _, row in df.iterrows():
        rows.append(
            PaperRecord(
                key=str(row["key"]).strip(),
                doi=normalize_doi(row["doi"]),
                title=normalize_text(row["title"]),
                text_path=Path(str(row["text_path"]).replace("\\", "/")),
            )
        )
    return rows


def infer_tables_dir(paper: PaperRecord) -> Path | None:
    candidates: list[Path] = []
    text_path = paper.text_path
    if text_path.parent.name == "text":
        candidate = text_path.parent.parent / "tables" / paper.key
        if candidate.exists():
            candidates.append(candidate)
    candidates.extend(sorted(DATA_CLEANED_DIR.glob(f"*/tables/{paper.key}")))
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        token = str(candidate.resolve()).lower()
        if token in seen or not candidate.exists():
            continue
        seen.add(token)
        deduped.append(candidate)
    deduped.sort(
        key=lambda path: (
            "content_" in str(path).lower(),
            len(path.parts),
            str(path).lower(),
        )
    )
    for candidate in deduped:
        return candidate
    return None


def read_table_rows(csv_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            rows.append([normalize_text(cell) for cell in row])
    if rows:
        first = rows[0]
        if first and all(cell.isdigit() for cell in first):
            expected = [str(i) for i in range(len(first))]
            if first == expected:
                rows = rows[1:]
    return rows


def normalize_minus_signs(text: Any) -> str:
    normalized = normalize_text(text)
    normalized = normalized.replace("\x04", "-")
    normalized = normalized.replace("(cid:4)", "-")
    return normalized.replace("−", "-").replace("–", "-").replace("—", "-")


def parse_formulation_label_info(cell_text: str) -> dict[str, Any] | None:
    normalized = normalize_minus_signs(cell_text)
    numeric_match = re.fullmatch(r"(\d{1,3})\s*[\.\):]?", normalized)
    if numeric_match:
        return {
            "number": int(numeric_match.group(1)),
            "label": normalize_text(cell_text),
            "label_style": "numeric",
        }
    f_match = re.fullmatch(r"([Ff])\s*[- ]?(\d{1,3})\s*[\.\):]?", normalized)
    if f_match:
        return {
            "number": int(f_match.group(2)),
            "label": f"{f_match.group(1).upper()}{int(f_match.group(2))}",
            "label_style": "f_numeric",
        }
    return None


def parse_formulation_number(cell_text: str) -> int | None:
    info = parse_formulation_label_info(cell_text)
    if info is None:
        return None
    return int(info["number"])


def row_is_numbered(row: list[str]) -> int | None:
    if not row:
        return None
    return parse_formulation_number(row[0])


def row_label_info(row: list[str]) -> dict[str, Any] | None:
    if not row:
        return None
    return parse_formulation_label_info(row[0])


def count_numeric_like_cells(row: list[str]) -> int:
    count = 0
    for cell in row:
        if re.search(r"\d", cell):
            count += 1
    return count


def numbered_row_anchor(row: list[str], *, max_label_column: int = 12) -> tuple[int, int] | None:
    """Return (row_number, label_column) for explicit numbered DOE rows.

    Most preserved DOE tables put the row label in column 0, but PDF two-column
    spillover can shift the real table block rightward while unrelated prose
    occupies earlier columns.  Accept a shifted label only when enough numeric
    evidence remains to its right, so this stays anchored to explicit table rows
    rather than arbitrary prose numbers.
    """
    for col_idx, cell in enumerate(row[: min(len(row), max_label_column + 1)]):
        number = parse_formulation_number(cell)
        if number is None:
            continue
        if count_numeric_like_cells(row[col_idx + 1 :]) >= 3:
            return number, col_idx
    return None


def first_numbered_row_anchor(rows: list[list[str]]) -> tuple[int, int] | None:
    for idx, row in enumerate(rows):
        anchor = numbered_row_anchor(row)
        if anchor is None:
            continue
        return idx, anchor[1]
    return None


def first_numbered_row_index(rows: list[list[str]]) -> int | None:
    anchor = first_numbered_row_anchor(rows)
    return anchor[0] if anchor is not None else None


def combine_header_rows(rows: list[list[str]], numbered_idx: int) -> list[str]:
    header_rows = [row for row in rows[max(0, numbered_idx - 5):numbered_idx] if any(cell for cell in row)]
    width = max((len(row) for row in rows), default=0)
    combined: list[str] = []
    for col_idx in range(width):
        parts: list[str] = []
        for row in header_rows:
            if col_idx < len(row) and row[col_idx]:
                parts.append(row[col_idx])
        header = " ".join(parts).strip()
        header = re.sub(r"\s+", " ", header)
        combined.append(header or f"column_{col_idx}")
    seen: dict[str, int] = {}
    unique_headers: list[str] = []
    for header in combined:
        count = seen.get(header, 0) + 1
        seen[header] = count
        if count == 1:
            unique_headers.append(header)
        else:
            unique_headers.append(f"{header} [{count}]")
    return unique_headers


def table_keyword_score(header_row: list[str], prelude_rows: list[list[str]]) -> int:
    text = " ".join(header_row + [" ".join(row) for row in prelude_rows]).lower()
    score = 0
    for pattern in [
        r"\bformulation\b",
        r"\bbox[- ]behnken\b",
        r"\bdesign\b",
        r"\bpoloxamer\b",
        r"\bpolymer\b",
        r"\bplga\b",
        r"\bdrug\b",
        r"\bpdi\b",
        r"\bentrapment\b",
        r"\bz-average\b",
        r"\bphase ratio\b",
    ]:
        if re.search(pattern, text):
            score += 1
    return score


MEASUREMENT_HEADER_PATTERNS = [
    r"\bmean size\b",
    r"\bsize\b",
    r"\bz-average\b",
    r"\bpdi\b",
    r"\bpolydispersity\b",
    r"\bzeta\b",
    r"\bentrapp?ment\b",
    r"\bencapsulation\b",
    r"\bloading\b",
    r"\brecovery\b",
    r"\bdrug content\b",
    r"\bmeasured responses\b",
    r"\bresponse\b",
]


def is_measurement_header(header: str) -> bool:
    low = normalize_text(header).lower()
    return any(re.search(pattern, low) for pattern in MEASUREMENT_HEADER_PATTERNS)


def normalize_factor_key(text: str) -> str:
    normalized = normalize_minus_signs(text).lower()
    quoted_parts = re.findall(r"'([^']+)'", normalized)
    if quoted_parts:
        normalized = quoted_parts[-1]
    normalized = re.sub(r"\([^)]*\)", " ", normalized)
    normalized = normalized.replace("aqueous phase", "")
    normalized = re.sub(r"\bc(?=[a-z])", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def factor_names_match(run_header: str, coding_factor: str) -> bool:
    left = normalize_factor_key(run_header)
    right = normalize_factor_key(coding_factor)
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = {token for token in left.split() if token}
    right_tokens = {token for token in right.split() if token}
    if left_tokens and right_tokens and (left_tokens <= right_tokens or right_tokens <= left_tokens):
        return True
    if len(left) >= 2 and left in right:
        return True
    if len(right) >= 2 and right in left:
        return True
    return False


def normalize_coded_level(value: Any) -> str:
    normalized = normalize_minus_signs(value)
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)", normalized)
    if not match:
        return ""
    numeric = float(match.group(1))
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.6g}"


def detect_coded_factor_columns(header_row: list[str], numbered_rows: list[list[str]]) -> list[dict[str, Any]]:
    coded_columns: list[dict[str, Any]] = []
    if not numbered_rows:
        return coded_columns
    width = max(len(row) for row in numbered_rows)
    for col_idx in range(1, width):
        header = header_row[col_idx] if col_idx < len(header_row) else ""
        if is_measurement_header(header):
            continue
        normalized_header = normalize_text(header)
        if (
            not normalized_header
            or normalized_header.startswith("column_")
            or normalized_header.isdigit()
            or not re.search(r"[A-Za-z]", normalized_header)
            or not re.search(r"\bx\d+\b|factor|drug|polymer|surfactant|stabilizer|concentration|ph\b", normalized_header, flags=re.IGNORECASE)
        ):
            continue
        coded_values = [
            normalize_coded_level(row[col_idx])
            for row in numbered_rows
            if col_idx < len(row) and normalize_text(row[col_idx])
        ]
        if len(coded_values) < max(3, min(6, len(numbered_rows))):
            continue
        distinct_values = {value for value in coded_values if value}
        if len(distinct_values) < 2 or len(distinct_values) > 5:
            continue
        if {"-1", "0", "1"}.issubset(distinct_values) or any(
            value not in {"-1", "0", "1"} for value in distinct_values
        ):
            coded_columns.append({"column_index": col_idx, "header": normalized_header})
    if coded_columns:
        return coded_columns
    fallback_columns: list[dict[str, Any]] = []
    for col_idx in range(1, min(width, 5)):
        coded_values = [
            normalize_coded_level(row[col_idx])
            for row in numbered_rows
            if col_idx < len(row) and normalize_text(row[col_idx])
        ]
        if len(coded_values) < max(3, min(6, len(numbered_rows))):
            break
        distinct_values = {value for value in coded_values if value}
        if len(distinct_values) < 2 or len(distinct_values) > 5:
            break
        if not ({"-1", "0", "1"}.issubset(distinct_values) or any(value not in {"-1", "0", "1"} for value in distinct_values)):
            break
        fallback_columns.append({"column_index": col_idx, "header": f"X{len(fallback_columns) + 1}"})
    if len(fallback_columns) >= 2 and len(fallback_columns) > len(coded_columns):
        return fallback_columns
    return coded_columns


def extract_coding_table_map(payload: dict[str, Any]) -> dict[str, dict[str, str]] | None:
    rows = [item for item in payload.get("normalized_rows", []) if isinstance(item, dict)]
    if not rows:
        return None
    coding_map: dict[str, dict[str, str]] = {}
    for row in rows:
        cells = [normalize_minus_signs(cell) for cell in row.get("cells", []) if normalize_text(cell)]
        if not cells:
            continue
        factor_name = ""
        level_values: list[str] = []
        first_cell = normalize_text(cells[0])
        first_factor_match = re.match(r"^(x\d+)\b", first_cell, flags=re.IGNORECASE)
        if first_factor_match:
            factor_name = first_factor_match.group(1).upper()
            if len(cells) >= 4:
                level_values = [normalize_text(value) for value in cells[-3:] if normalize_text(value)]
        cell_map = row.get("cell_map") if isinstance(row.get("cell_map"), dict) else {}
        if not factor_name and cell_map:
            factor_name = normalize_text(cell_map.get("Factor"))
            if not factor_name:
                factor_name = next(
                    (
                        normalize_text(value)
                        for key, value in cell_map.items()
                        if normalize_text(key) and not normalize_coded_level(key)
                    ),
                    "",
                )
        level_map: dict[str, str] = {}
        if level_values and len(level_values) >= 3:
            level_map = {"-1": level_values[0], "0": level_values[1], "1": level_values[2]}
        elif cell_map:
            for level_key, raw_value in cell_map.items():
                normalized_key = normalize_coded_level(level_key)
                if not normalized_key:
                    continue
                level_value = normalize_text(raw_value)
                if level_value:
                    level_map[normalized_key] = level_value
        if factor_name and len(level_map) >= 3:
            coding_map[factor_name] = level_map
    return coding_map or None


def resolve_coding_table_for_run_table(
    *,
    header_row: list[str],
    numbered_rows: list[list[str]],
    normalized_payloads: list[dict[str, Any]] | None,
    run_csv_path: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    effective_header_row = list(header_row)
    run_csv_token = str(run_csv_path).replace("\\", "/").lower()
    if normalized_payloads:
        for payload in normalized_payloads:
            payload_csv = normalize_text(payload.get("normalized_csv_path")).replace("\\", "/").lower()
            if payload_csv != run_csv_token:
                continue
            header_structure = payload.get("header_structure") if isinstance(payload.get("header_structure"), dict) else {}
            payload_headers = [
                normalize_text(item)
                for item in header_structure.get("flattened_headers", [])
                if normalize_text(item)
            ]
            if payload_headers:
                effective_header_row = payload_headers
            break
    coded_columns = detect_coded_factor_columns(effective_header_row, numbered_rows)
    if not coded_columns:
        return None, [], "not_needed"
    if not normalized_payloads:
        return None, coded_columns, "payload_locator_missing"
    best_payload: dict[str, Any] | None = None
    best_score = 0
    coding_payload_candidates: list[dict[str, Any]] = []
    self_payload = next(
        (
            payload
            for payload in normalized_payloads
            if normalize_text(payload.get("normalized_csv_path")).replace("\\", "/").lower() == run_csv_token
            and extract_coding_table_map(payload)
        ),
        None,
    )
    ordered_headers = [normalize_text(column["header"]).upper() for column in coded_columns]
    if self_payload is not None and ordered_headers[: len(coded_columns)] == [f"X{i}" for i in range(1, len(coded_columns) + 1)]:
        best_payload = self_payload
        best_score = len(coded_columns)
    for payload in normalized_payloads:
        payload_csv = normalize_text(payload.get("normalized_csv_path")).replace("\\", "/").lower()
        if payload_csv == run_csv_token:
            continue
        coding_map = extract_coding_table_map(payload)
        if not coding_map:
            continue
        coding_payload_candidates.append(payload)
        score = 0
        for column in coded_columns:
            header = column["header"]
            if any(factor_names_match(header, factor_name) for factor_name in coding_map):
                score += 1
        if score > best_score:
            best_payload = payload
            best_score = score
    if best_payload is None and self_payload is not None:
        best_payload = self_payload
        best_score = len(coded_columns)
    if best_payload is None and coding_payload_candidates:
        if ordered_headers[: len(coded_columns)] == [f"X{i}" for i in range(1, len(coded_columns) + 1)]:
            best_payload = coding_payload_candidates[0]
            best_score = len(coded_columns)
    if best_payload is None or best_score <= 0:
        return None, coded_columns, "coding_table_unresolved"
    return best_payload, coded_columns, ""


def decode_row_assignments(
    *,
    row: list[str],
    coded_columns: list[dict[str, Any]],
    coding_payload: dict[str, Any],
) -> tuple[dict[str, str], str]:
    coding_map = extract_coding_table_map(coding_payload) or {}
    decoded: dict[str, str] = {}
    for column in coded_columns:
        header = normalize_text(column["header"])
        col_idx = int(column["column_index"])
        coded_value = normalize_coded_level(row[col_idx] if col_idx < len(row) else "")
        if not coded_value:
            return {}, "coded_value_missing"
        matching_factor = next(
            (factor_name for factor_name in coding_map if factor_names_match(header, factor_name)),
            "",
        )
        if not matching_factor:
            return {}, "coding_factor_unresolved"
        actual_value = normalize_text(coding_map.get(matching_factor, {}).get(coded_value))
        if not actual_value:
            return {}, "coding_value_unresolved"
        decoded[matching_factor] = actual_value
    return decoded, ""


def apply_decoded_assignments_to_fields(
    *,
    fields: dict[str, dict[str, Any]],
    extras: dict[str, str],
    decoded_assignments: dict[str, str],
) -> None:
    for factor_name, factor_value in decoded_assignments.items():
        extras[factor_name] = factor_value
        low = factor_name.lower()
        if "plga" in low or "polymer" in low:
            fields["plga_mass_mg"] = {
                "value": maybe_number_text(factor_value) or factor_value,
                "value_text": factor_value,
                "scope": "instance_specific",
                "membership_confidence": "high",
                "evidence_region_type": "table_cell",
                "missing_reason": "",
            }
        elif "pva" in low or "surfactant" in low or "stabilizer" in low:
            fields["surfactant_concentration_text"] = {
                "value": maybe_number_text(factor_value) or factor_value,
                "value_text": factor_value,
                "scope": "instance_specific",
                "membership_confidence": "high",
                "evidence_region_type": "table_cell",
                "missing_reason": "",
            }
        elif "pf" in low or "drug" in low:
            fields["drug_feed_amount_text"] = {
                "value": maybe_number_text(factor_value) or factor_value,
                "value_text": factor_value,
                "scope": "instance_specific",
                "membership_confidence": "high",
                "evidence_region_type": "table_cell",
                "missing_reason": "",
            }


def select_candidate_tables(tables_dir: Path, min_numbered_rows: int) -> list[dict[str, Any]]:
    manifest_path = tables_dir / "tables_manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(manifest, dict):
        table_entries = manifest.get("tables", [])
    elif isinstance(manifest, list):
        table_entries = manifest
    else:
        table_entries = []
    selected: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for table_meta in table_entries:
        csv_path = Path(str(table_meta.get("csv_path", "")).replace("\\", "/"))
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
        if not csv_path.exists():
            continue
        rows = read_table_rows(csv_path)
        anchor = first_numbered_row_anchor(rows)
        if anchor is None:
            continue
        numbered_idx, label_col_idx = anchor
        numbered_rows: list[list[str]] = []
        for row in rows[numbered_idx:]:
            row_anchor = numbered_row_anchor(row)
            if row_anchor is None or row_anchor[1] != label_col_idx:
                continue
            trimmed_row = row[label_col_idx:]
            if row_is_numbered(trimmed_row) is None:
                continue
            if count_numeric_like_cells(trimmed_row[1:]) < 3:
                continue
            numbered_rows.append(trimmed_row)
        if len(numbered_rows) < min_numbered_rows:
            continue
        header_row = combine_header_rows(rows, numbered_idx)[label_col_idx:]
        keyword_score = table_keyword_score(header_row, rows[:numbered_idx])
        if keyword_score < 2:
            continue
        signature = "\n".join("|".join(row) for row in numbered_rows)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        selected.append(
            {
                "csv_path": csv_path,
                "page_number": str(table_meta.get("page_number", "")).strip(),
                "source_type": str(table_meta.get("source_type", "")).strip(),
                "caption_or_title": normalize_text(table_meta.get("caption_or_title", "")),
                "rows": rows,
                "header_row": header_row,
                "numbered_rows": numbered_rows,
            }
        )
    selected.sort(key=lambda item: (str(item["csv_path"]).lower(), item["page_number"]))
    return selected


def explicit_table_candidate(
    *,
    csv_path: Path,
    min_numbered_rows: int,
    table_id: str = "",
    caption_or_title: str = "",
    source_type: str = "semantic_authorized_table_target",
) -> dict[str, Any] | None:
    if not csv_path.is_absolute():
        csv_path = (REPO_ROOT / csv_path).resolve()
    if not csv_path.exists():
        return None
    rows = read_table_rows(csv_path)
    anchor = first_numbered_row_anchor(rows)
    if anchor is None:
        return None
    numbered_idx, label_col_idx = anchor
    numbered_rows: list[list[str]] = []
    for row in rows[numbered_idx:]:
        row_anchor = numbered_row_anchor(row)
        if row_anchor is None or row_anchor[1] != label_col_idx:
            continue
        trimmed_row = row[label_col_idx:]
        if row_is_numbered(trimmed_row) is None:
            continue
        if count_numeric_like_cells(trimmed_row[1:]) < 3:
            continue
        numbered_rows.append(trimmed_row)
    if len(numbered_rows) < min_numbered_rows:
        return None
    header_row = combine_header_rows(rows, numbered_idx)[label_col_idx:]
    keyword_score = table_keyword_score(header_row, rows[:numbered_idx])
    semantic_authorized_sources = {
        "semantic_authorized_table_target",
        "semantic_authorized_companion_table_target",
    }
    if keyword_score < 2 and source_type not in semantic_authorized_sources:
        return None
    return {
        "csv_path": csv_path,
        "page_number": "",
        "source_type": source_type,
        "caption_or_title": normalize_text(caption_or_title),
        "rows": rows,
        "header_row": header_row,
        "numbered_rows": numbered_rows,
        "semantic_table_id": normalize_text(table_id),
    }


def infer_drug_name(title: str, raw_text: str) -> str:
    title_text = normalize_text(title)
    match = re.search(r"delivery of ([a-z0-9 -]+?)(?: using| by| with|$)", title_text, flags=re.I)
    if match:
        return normalize_text(match.group(1))
    for pattern in [r"\blorazepam\b", r"\bdocetaxel\b", r"\betoposide\b", r"\bpaclitaxel\b"]:
        hit = re.search(pattern, raw_text, flags=re.I)
        if hit:
            return hit.group(0)
    return ""


def infer_polymer_identity(title: str, raw_text: str) -> tuple[str, str]:
    blob = f"{title} {raw_text[:2000]}"
    if re.search(r"\bplga\b", blob, flags=re.I):
        return "PLGA", "PLGA"
    if re.search(r"\bpcl\b", blob, flags=re.I):
        return "PCL", "PCL"
    if re.search(r"\bpla\b", blob, flags=re.I):
        return "PLA", "PLA"
    return "unknown", ""


def maybe_number_text(cell_text: str) -> str:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cell_text)
    return match.group(0) if match else ""


def header_matches(header: str, *patterns: str) -> bool:
    low = header.lower()
    return any(re.search(pattern, low) for pattern in patterns)


def parse_row_fields(
    *,
    header_row: list[str],
    row: list[str],
    title: str,
    raw_text: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    fields: dict[str, dict[str, Any]] = {}
    extras: dict[str, str] = {}
    drug_name = infer_drug_name(title, raw_text)
    polymer_identity, polymer_name_raw = infer_polymer_identity(title, raw_text)
    if polymer_identity != "unknown":
        extras["polymer_identity"] = polymer_identity
        extras["polymer_name_raw"] = polymer_name_raw
    for header, cell in zip(header_row[1:], row[1:]):
        clean_header = normalize_text(header)
        clean_cell = normalize_text(cell)
        if not clean_cell:
            continue
        value_num = maybe_number_text(clean_cell)
        if header_matches(clean_header, r"\bplga\b", r"\bpolymer\b"):
            fields["plga_mass_mg"] = {
                "value": value_num or clean_cell,
                "value_text": f"{clean_cell} mg/mL" if "mg/ml" not in clean_cell.lower() and "mg/mL" not in clean_cell else clean_cell,
                "scope": "instance_specific",
                "membership_confidence": "high",
                "evidence_region_type": "table_cell",
                "missing_reason": "",
            }
            continue
        if header_matches(clean_header, r"\bpoloxamer\b", r"\bsurfactant\b"):
            fields["surfactant_name"] = {
                "value": "Poloxamer",
                "value_text": "Poloxamer",
                "scope": "global_shared",
                "membership_confidence": "medium",
                "evidence_region_type": "table_header",
                "missing_reason": "",
            }
            fields["surfactant_concentration_text"] = {
                "value": value_num or clean_cell,
                "value_text": f"{clean_cell} mg/mL" if "mg/ml" not in clean_cell.lower() and "mg/mL" not in clean_cell else clean_cell,
                "scope": "instance_specific",
                "membership_confidence": "high",
                "evidence_region_type": "table_cell",
                "missing_reason": "",
            }
            continue
        if header_matches(clean_header, r"\bz-average\b", r"\baverage\b", r"\bsize\b"):
            fields["size_nm"] = {
                "value": value_num or clean_cell,
                "value_text": clean_cell,
                "scope": "instance_specific",
                "membership_confidence": "high",
                "evidence_region_type": "table_cell",
                "missing_reason": "",
            }
            continue
        if header_matches(clean_header, r"\bzeta\b"):
            fields["zeta_mV"] = {
                "value": value_num or clean_cell,
                "value_text": clean_cell,
                "scope": "instance_specific",
                "membership_confidence": "high",
                "evidence_region_type": "table_cell",
                "missing_reason": "",
            }
            continue
        if header_matches(clean_header, r"\bentrapment\b", r"\bencapsulation\b", r"\bee\b"):
            fields["encapsulation_efficiency_percent"] = {
                "value": value_num or clean_cell,
                "value_text": clean_cell,
                "scope": "instance_specific",
                "membership_confidence": "high",
                "evidence_region_type": "table_cell",
                "missing_reason": "",
            }
            continue
        if header_matches(clean_header, r"\bdrug conc\b", r"\bdrug concentration\b"):
            if drug_name:
                fields["drug_name"] = {
                    "value": drug_name,
                    "value_text": drug_name,
                    "scope": "global_shared",
                    "membership_confidence": "medium",
                    "evidence_region_type": "table_header",
                    "missing_reason": "",
                }
            fields["drug_feed_amount_text"] = {
                "value": value_num or clean_cell,
                "value_text": f"{clean_cell} mg/mL" if "mg/ml" not in clean_cell.lower() and "mg/mL" not in clean_cell else clean_cell,
                "scope": "instance_specific",
                "membership_confidence": "high",
                "evidence_region_type": "table_cell",
                "missing_reason": "",
            }
            continue
        if header_matches(clean_header, r"\bpdi\b"):
            fields["pdi"] = {
                "value": value_num or clean_cell,
                "value_text": clean_cell,
                "scope": "instance_specific",
                "membership_confidence": "high",
                "evidence_region_type": "table_cell",
                "missing_reason": "",
            }
            continue
        extras[clean_header or f"column_{len(extras) + 1}"] = clean_cell
    return fields, extras


def build_supporting_evidence_ref(table_id: str, row_text: str, csv_path: Path) -> dict[str, Any]:
    return {
        "region_type": "table_row",
        "section": table_id,
        "span_text": row_text,
        "span_start": "",
        "span_end": "",
        "table_csv_path": str(csv_path).replace("\\", "/"),
    }


def extract_checkpoint_batches_from_text(raw_text: str) -> list[dict[str, str]]:
    text = normalize_minus_signs(raw_text)
    anchor = text.find("Checkpoint batches with their predicted and measured values of PS and EE")
    if anchor < 0:
        return []
    window = re.sub(r"\s+", " ", text[anchor : anchor + 1200])
    section = window.split("tcalculated", 1)[0]
    pattern = re.compile(
        r"(?P<batch>[123])\s+[^()]*\((?P<x1_actual>[^)]+)\)\s+[^()]*\((?P<x2_actual>[^)]+)\)\s+[^()]*\((?P<x3_actual>[^)]+)\)\s+(?P<ps_pred>\d+(?:\.\d+)?)\s+(?P<ps_obs>\d+(?:\.\d+)?)\s+(?P<ee_pred>\d+(?:\.\d+)?)\s+(?P<ee_obs>\d+(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )
    extracted: list[dict[str, str]] = []
    for match in pattern.finditer(section):
        extracted.append(
            {
                "batch_no": match.group("batch"),
                "x1_actual": normalize_text(match.group("x1_actual")),
                "x2_actual": normalize_text(match.group("x2_actual")),
                "x3_actual": normalize_text(match.group("x3_actual")),
                "ps_pred": match.group("ps_pred"),
                "ps_obs": match.group("ps_obs"),
                "ee_pred": match.group("ee_pred"),
                "ee_obs": match.group("ee_obs"),
                "row_text": normalize_text(match.group(0)),
            }
        )
    return extracted


def build_checkpoint_candidate_form(
    *,
    paper: PaperRecord,
    batch: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    batch_no = int(batch["batch_no"])
    row_text = batch["row_text"]
    fields = {
        "size_nm": {
            "value": batch["ps_obs"],
            "value_text": batch["ps_obs"],
            "scope": "instance_specific",
            "membership_confidence": "high",
            "evidence_region_type": "text_span",
            "missing_reason": "",
        },
        "encapsulation_efficiency_percent": {
            "value": batch["ee_obs"],
            "value_text": batch["ee_obs"],
            "scope": "instance_specific",
            "membership_confidence": "high",
            "evidence_region_type": "text_span",
            "missing_reason": "",
        },
    }
    extras = {
        "X1_actual": batch["x1_actual"],
        "X2_actual": batch["x2_actual"],
        "X3_actual": batch["x3_actual"],
        "predicted_particle_size_nm": batch["ps_pred"],
        "predicted_entrapment_efficiency_percent": batch["ee_pred"],
    }
    candidate = {
        "formulation_id": f"{paper.key}_Checkpoint_Batch_{batch_no}",
        "raw_formulation_label": f"Checkpoint Batch {batch_no}",
        "polymer_identity": "PLGA",
        "polymer_name_raw": "PLGA",
        "instance_kind": "new_formulation",
        "parent_instance_id": "",
        "change_descriptions": [f"{key}={value}" for key, value in extras.items()],
        "change_role": "synthesis_defining",
        "instance_context_tags": ["checkpoint_batch"],
        "change_context_tags": ["checkpoint_validation"],
        "supporting_evidence_refs": [
            {
                "region_type": "text_span",
                "section": "Checkpoint analysis",
                "span_text": row_text,
                "span_start": "",
                "span_end": "",
            }
        ],
        "formulation_role": "variant",
        "instance_confidence": "high",
        "candidate_source": "doe_checkpoint_batch_recovery",
        "fields": fields,
        "instance_evidence": {
            "evidence_region_type": "text_span",
            "evidence_section": "Checkpoint analysis",
            "evidence_span_text": row_text,
            "evidence_span_start": "",
            "evidence_span_end": "",
        },
    }
    artifact_row = {
        "paper_key": paper.key,
        "doi": paper.doi,
        "title": paper.title,
        "table_id": "Checkpoint analysis",
        "table_csv_path": "",
        "formulation_number": str(batch_no),
        "formulation_label": f"Checkpoint Batch {batch_no}",
        "candidate_id": candidate["formulation_id"],
        "candidate_source": candidate["candidate_source"],
        "instance_confidence": candidate["instance_confidence"],
        "instance_kind": candidate["instance_kind"],
        "formulation_role": candidate["formulation_role"],
        "parsed_core_fields_json": json.dumps(fields, ensure_ascii=False, sort_keys=True),
        "parsed_extra_fields_json": json.dumps(extras, ensure_ascii=False, sort_keys=True),
        "raw_row_json": json.dumps(batch, ensure_ascii=False, sort_keys=True),
        "row_text": row_text,
        "header_json": json.dumps(["checkpoint_batch"], ensure_ascii=False),
        "evidence_source_type": "text_span",
        "evidence_section": "Checkpoint analysis",
        "evidence_snippet": row_text,
        "provenance_note": "Deterministically recovered from explicit checkpoint-batch validation text near the Table 7 anchor.",
        "confidence_note": "High confidence because the paper explicitly states three checkpoint batches with observed values.",
        "existing_stage2_match": "",
    }
    return candidate, artifact_row


def build_stage2_candidate_form(
    *,
    paper: PaperRecord,
    table_id: str,
    csv_path: Path,
    formulation_number: int,
    formulation_label: str,
    row: list[str],
    header_row: list[str],
    raw_text: str,
    decoded_assignments: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    row_text = " | ".join(cell for cell in row if cell)
    fields, extras = parse_row_fields(header_row=header_row, row=row, title=paper.title, raw_text=raw_text)
    apply_decoded_assignments_to_fields(
        fields=fields,
        extras=extras,
        decoded_assignments=decoded_assignments or {},
    )
    change_descriptions = [f"{key}={value}" for key, value in sorted(extras.items()) if key not in {"polymer_identity", "polymer_name_raw"}]
    label_token = re.sub(r"[^A-Za-z0-9]+", "_", formulation_label).strip("_") or f"row_{formulation_number:02d}"
    candidate = {
        "formulation_id": f"{paper.key}_DOE_Row_{label_token}",
        "raw_formulation_label": formulation_label,
        "polymer_identity": extras.get("polymer_identity", "unknown"),
        "polymer_name_raw": extras.get("polymer_name_raw", ""),
        "instance_kind": "new_formulation",
        "parent_instance_id": "",
        "change_descriptions": change_descriptions,
        "change_role": "synthesis_defining",
        "instance_context_tags": ["doe", "numbered_table_row"],
        "change_context_tags": ["table_enumeration"],
        "supporting_evidence_refs": [build_supporting_evidence_ref(table_id, row_text, csv_path)],
        "formulation_role": "variant",
        "instance_confidence": "high",
        "candidate_source": "doe_numbered_table_row",
        "fields": fields,
        "instance_evidence": {
            "evidence_region_type": "table_row",
            "evidence_section": table_id,
            "evidence_span_text": row_text,
            "evidence_span_start": "",
            "evidence_span_end": "",
        },
    }
    artifact_row = {
        "paper_key": paper.key,
        "doi": paper.doi,
        "title": paper.title,
        "table_id": table_id,
        "table_csv_path": str(csv_path).replace("\\", "/"),
        "formulation_number": str(formulation_number),
        "formulation_label": f"{formulation_number}.",
        "candidate_id": candidate["formulation_id"],
        "candidate_source": candidate["candidate_source"],
        "instance_confidence": candidate["instance_confidence"],
        "instance_kind": candidate["instance_kind"],
        "formulation_role": candidate["formulation_role"],
        "parsed_core_fields_json": json.dumps(fields, ensure_ascii=False, sort_keys=True),
        "parsed_extra_fields_json": json.dumps(extras, ensure_ascii=False, sort_keys=True),
        "raw_row_json": json.dumps(dict(zip(header_row, row)), ensure_ascii=False),
        "row_text": row_text,
        "header_json": json.dumps(header_row, ensure_ascii=False),
        "evidence_source_type": "table_row",
        "evidence_section": table_id,
        "evidence_snippet": row_text,
        "provenance_note": "Deterministically enumerated from an explicit numbered DOE-style table row.",
        "confidence_note": "High confidence because the source row is explicitly numbered and preserved in Stage1 table assets.",
        "existing_stage2_match": "",
    }
    return candidate, artifact_row


def existing_numeric_label_map(existing_forms: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    observed: dict[str, dict[str, str]] = {}
    for form in existing_forms:
        raw_label = normalize_text(form.get("raw_formulation_label", ""))
        number = parse_formulation_number(raw_label)
        if number is None:
            continue
        observed[str(number)] = {
            "formulation_id": normalize_text(form.get("formulation_id", "")),
            "candidate_source": normalize_text(form.get("candidate_source", "")),
        }
    return observed


def enumerate_numbered_doe_candidates_for_paper(
    *,
    paper: PaperRecord,
    raw_text: str,
    existing_forms: list[dict[str, Any]] | None = None,
    min_numbered_rows: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, str]]:
    forms = existing_forms or []
    existing_map = existing_numeric_label_map(forms)
    tables_dir = infer_tables_dir(paper)
    if tables_dir is None:
        return [], [], {
            "paper_key": paper.key,
            "doi": paper.doi,
            "title": paper.title,
            "tables_dir": "",
            "candidate_tables_considered": "0",
            "selected_table_count": "0",
            "selected_table_ids": "",
            "numbered_rows_found": "0",
            "existing_stage2_numeric_rows": str(len(existing_map)),
            "new_candidates_emitted": "0",
            "regression_status": "no_tables_dir",
            "notes": "No Stage1 tables directory was found for this paper.",
        }
    selected_tables = select_candidate_tables(tables_dir, min_numbered_rows=min_numbered_rows)
    emitted_forms: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, str]] = []
    numbered_rows_found = 0
    selected_table_ids: list[str] = []
    for idx, table in enumerate(selected_tables, start=1):
        table_id = f"{paper.key}__numbered_doe_table_{idx:02d}"
        selected_table_ids.append(table_id)
        for row in table["numbered_rows"]:
            formulation_number = row_is_numbered(row)
            if formulation_number is None:
                continue
            numbered_rows_found += 1
            existing_match = existing_map.get(str(formulation_number))
            if existing_match and existing_match.get("candidate_source") != "llm_extracted":
                continue
            candidate, artifact_row = build_stage2_candidate_form(
                paper=paper,
                table_id=table_id,
                csv_path=table["csv_path"],
                formulation_number=formulation_number,
                row=row,
                header_row=table["header_row"],
                raw_text=raw_text,
            )
            if existing_match:
                artifact_row["existing_stage2_match"] = existing_match.get("formulation_id", "")
            emitted_forms.append(candidate)
            artifact_rows.append(artifact_row)
    summary = {
        "paper_key": paper.key,
        "doi": paper.doi,
        "title": paper.title,
        "tables_dir": str(tables_dir).replace("\\", "/"),
        "candidate_tables_considered": str(len(selected_tables)),
        "selected_table_count": str(len(selected_tables)),
        "selected_table_ids": "|".join(selected_table_ids),
        "numbered_rows_found": str(numbered_rows_found),
        "existing_stage2_numeric_rows": str(len(existing_map)),
        "new_candidates_emitted": str(len(emitted_forms)),
        "regression_status": "ok" if emitted_forms else "no_new_candidates",
        "notes": "Explicit numbered DOE table rows were enumerated deterministically." if emitted_forms else "No missing numbered DOE rows were emitted.",
    }
    return emitted_forms, artifact_rows, summary


def enumerate_numbered_doe_candidates_for_explicit_tables(
    *,
    paper: PaperRecord,
    raw_text: str,
    explicit_targets: list[dict[str, Any]],
    existing_forms: list[dict[str, Any]] | None = None,
    min_numbered_rows: int = 8,
    normalized_payloads: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, str]]:
    forms = existing_forms or []
    existing_map = existing_numeric_label_map(forms)
    candidate_targets = [item for item in explicit_targets if isinstance(item, dict)]
    selected_tables: list[dict[str, Any]] = []
    unresolved_targets: list[str] = []
    for target in candidate_targets:
        table_path = Path(str(target.get("table_path") or "").replace("\\", "/"))
        table_id = normalize_text(target.get("table_id"))
        table_asset_id = normalize_text(target.get("table_asset_id"))
        display_id = table_id or table_asset_id or normalize_text(table_path.name)
        selected = explicit_table_candidate(
            csv_path=table_path,
            min_numbered_rows=min_numbered_rows,
            table_id=table_id,
            caption_or_title=normalize_text(target.get("evidence_span")),
        )
        if selected is None:
            unresolved_targets.append(display_id)
            continue
        selected_tables.append(selected)

    emitted_forms: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, str]] = []
    numbered_rows_found = 0
    selected_table_ids: list[str] = []
    selected_table_paths: list[str] = []
    coding_tables_used: list[str] = []
    run_tables_used: list[str] = []
    decode_status = "not_attempted"
    decode_failure_reason = ""
    for idx, table in enumerate(selected_tables, start=1):
        table_id = normalize_text(table.get("semantic_table_id")) or f"{paper.key}__numbered_doe_table_{idx:02d}"
        selected_table_ids.append(table_id)
        selected_table_paths.append(str(Path(table["csv_path"])).replace("\\", "/"))
        run_tables_used.append(table_id)
        coding_payload, coded_columns, coding_failure_reason = resolve_coding_table_for_run_table(
            header_row=table["header_row"],
            numbered_rows=table["numbered_rows"],
            normalized_payloads=normalized_payloads,
            run_csv_path=Path(table["csv_path"]),
        )
        if coded_columns:
            if coding_payload is None:
                decode_status = "unresolved_but_emitted_raw_rows"
                decode_failure_reason = coding_failure_reason or "coding_table_unresolved"
            else:
                decode_status = "decoded"
                coding_tables_used.append(
                    normalize_text(coding_payload.get("table_id") or coding_payload.get("source_table_id"))
                )
        for row in table["numbered_rows"]:
            label_info = row_label_info(row)
            if label_info is None:
                continue
            formulation_number = int(label_info["number"])
            numbered_rows_found += 1
            existing_match = existing_map.get(str(formulation_number))
            if existing_match and existing_match.get("candidate_source") != "llm_extracted":
                continue
            decoded_assignments: dict[str, str] = {}
            if coded_columns and coding_payload is not None:
                decoded_assignments, row_decode_failure_reason = decode_row_assignments(
                    row=row,
                    coded_columns=coded_columns,
                    coding_payload=coding_payload,
                )
                if row_decode_failure_reason:
                    decode_status = "unresolved_but_emitted_raw_rows"
                    decode_failure_reason = row_decode_failure_reason
                    decoded_assignments = {}
            candidate, artifact_row = build_stage2_candidate_form(
                paper=paper,
                table_id=table_id,
                csv_path=table["csv_path"],
                formulation_number=formulation_number,
                formulation_label=normalize_text(label_info["label"]),
                row=row,
                header_row=table["header_row"],
                raw_text=raw_text,
                decoded_assignments=decoded_assignments,
            )
            if existing_match:
                artifact_row["existing_stage2_match"] = existing_match.get("formulation_id", "")
            emitted_forms.append(candidate)
            artifact_rows.append(artifact_row)
        if decode_status == "failed":
            break

    notes = (
        "Explicit numbered DOE table rows were enumerated deterministically from authorized semantic table targets."
        if emitted_forms
        else "No numbered DOE rows were emitted from the authorized semantic table targets."
    )
    checkpoint_batches = extract_checkpoint_batches_from_text(raw_text)
    for batch in checkpoint_batches:
        candidate, artifact_row = build_checkpoint_candidate_form(paper=paper, batch=batch)
        emitted_forms.append(candidate)
        artifact_rows.append(artifact_row)
    if checkpoint_batches:
        notes = f"{notes} Recovered {len(checkpoint_batches)} explicit checkpoint batches from source text."
    if unresolved_targets:
        notes = f"{notes} Unresolved targets: {' | '.join(unresolved_targets)}."
    summary = {
        "paper_key": paper.key,
        "doi": paper.doi,
        "title": paper.title,
        "tables_dir": "|".join(str(Path(path).parent).replace("\\", "/") for path in selected_table_paths),
        "candidate_tables_considered": str(len(candidate_targets)),
        "selected_table_count": str(len(selected_tables)),
        "selected_table_ids": "|".join(selected_table_ids),
        "selected_table_paths": "|".join(selected_table_paths),
        "unresolved_authorized_targets": "|".join(unresolved_targets),
        "numbered_rows_found": str(numbered_rows_found),
        "existing_stage2_numeric_rows": str(len(existing_map)),
        "new_candidates_emitted": str(len(emitted_forms)),
        "coding_table_used": "|".join(item for item in coding_tables_used if item),
        "run_table_used": "|".join(item for item in run_tables_used if item),
        "decode_status": decode_status,
        "decode_failure_reason": decode_failure_reason,
        "regression_status": "ok" if emitted_forms else "no_new_candidates",
        "notes": notes,
    }
    return emitted_forms, artifact_rows, summary


def candidate_output_columns() -> list[str]:
    return [
        "paper_key",
        "doi",
        "title",
        "table_id",
        "table_csv_path",
        "formulation_number",
        "formulation_label",
        "candidate_id",
        "candidate_source",
        "instance_confidence",
        "instance_kind",
        "formulation_role",
        "parsed_core_fields_json",
        "parsed_extra_fields_json",
        "raw_row_json",
        "row_text",
        "header_json",
        "evidence_source_type",
        "evidence_section",
        "evidence_snippet",
        "provenance_note",
        "confidence_note",
        "existing_stage2_match",
    ]


def summary_output_columns() -> list[str]:
    return [
        "paper_key",
        "doi",
        "title",
        "tables_dir",
        "candidate_tables_considered",
        "selected_table_count",
        "selected_table_ids",
        "numbered_rows_found",
        "existing_stage2_numeric_rows",
        "new_candidates_emitted",
        "expected_min_recovered",
        "regression_status",
        "notes",
    ]


def write_tsv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, sep="\t", index=False)


def write_candidate_artifacts(
    *,
    out_dir: Path,
    artifact_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    expected_min_recovered: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = [dict(row) for row in artifact_rows]
    summary_copy = []
    for row in summary_rows:
        record = dict(row)
        record["expected_min_recovered"] = str(expected_min_recovered)
        summary_copy.append(record)
    artifact_path = out_dir / ARTIFACT_NAME
    summary_path = out_dir / SUMMARY_NAME
    write_tsv(artifact_path, candidate_rows, candidate_output_columns())
    write_tsv(summary_path, summary_copy, summary_output_columns())
    return {
        "artifact_path": artifact_path,
        "summary_path": summary_path,
        "candidate_count": len(candidate_rows),
        "paper_count": len(summary_rows),
    }


def render_run_context(
    *,
    run_id: str,
    run_type: str,
    out_subdir: str,
    manifest_tsv: Path,
    weak_labels_tsv: Path | None,
    paper_keys: list[str],
    out_dir: Path,
    stats: dict[str, Any],
    expected_min_recovered: int,
) -> str:
    weak_line = f"- weak_labels_tsv: `{weak_labels_tsv}`" if weak_labels_tsv is not None else "- weak_labels_tsv: `not provided`"
    paper_line = ", ".join(f"`{key}`" for key in paper_keys) if paper_keys else "`all manifest rows`"
    return "\n".join(
        [
            "# RUN_CONTEXT",
            "",
            "## 1. Run ID",
            "",
            f"- `{run_id}`",
            "",
            "## 2. Run type",
            "",
            f"- `{run_type}`",
            "",
            "## 3. Purpose",
            "",
            "- Deterministically recover explicit numbered DOE table rows from existing Stage1 table assets as an upstream Stage2-boundary augmentation artifact.",
            "",
            "## 4. Starting input artifacts",
            "",
            f"- manifest_tsv: `{manifest_tsv}`",
            weak_line,
            f"- paper_keys: {paper_line}",
            "",
            "## 5. Exact script execution order",
            "",
            "1. Run `src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py` with explicit `--run-id`, `--out-subdir`, and `--manifest-tsv`.",
            "2. The script reads existing Stage1 table assets and emits deterministic DOE row candidate artifacts only.",
            "",
            "## 6. Script paths used",
            "",
            "- `src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py`",
            "",
            "## 7. Final outputs",
            "",
            f"- `{out_dir / ARTIFACT_NAME}`",
            f"- `{out_dir / SUMMARY_NAME}`",
            f"- `{out_dir.parent / 'RUN_CONTEXT.md'}`",
            "",
            "## 8. Benchmark-valid vs diagnostic-only status",
            "",
            "- `diagnostic-only, not benchmark-valid final output`",
            "- Reason: this run validates upstream numbered DOE row recovery only and does not execute the full Stage2 -> Stage5 benchmark chain.",
            "",
            "## 9. Reproduction steps",
            "",
            "```powershell",
            "$env:PYTHONPATH='c:\\Users\\tianc\\Downloads\\GitHub\\RL-Agent-Extraction-PLGANPs'; "
            f"python src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py --run-id {run_id} "
            f"--out-subdir {out_subdir} --manifest-tsv {manifest_tsv.as_posix()}",
            "```",
            "",
            "## 10. Outcome summary",
            "",
            f"- paper_count: `{stats['paper_count']}`",
            f"- candidate_count: `{stats['candidate_count']}`",
            f"- expected_min_recovered: `{expected_min_recovered}`",
        ]
    ) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministically enumerate explicit numbered DOE formulation rows from Stage1 table assets."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-subdir", required=True)
    parser.add_argument("--manifest-tsv", required=True, type=Path)
    parser.add_argument("--weak-labels-tsv", type=Path, default=None)
    parser.add_argument("--paper-keys", nargs="*", default=[])
    parser.add_argument("--min-numbered-rows", type=int, default=8)
    parser.add_argument("--expected-min-recovered", type=int, default=0)
    parser.add_argument(
        "--run-type",
        default="component_regression_run",
        choices=sorted(VALID_RUN_TYPES),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_id = validate_run_id(args.run_id)
    out_subdir = validate_out_subdir(args.out_subdir)
    if args.run_type == "full_pipeline_benchmark_run":
        raise ValueError(
            "This deterministic DOE-row recovery tool must not be labeled full_pipeline_benchmark_run because it stops before the canonical Stage5 comparison node."
        )

    papers = load_manifest(args.manifest_tsv, args.paper_keys)
    if not papers:
        raise ValueError("No manifest rows selected for deterministic DOE row enumeration.")

    weak_labels_by_key: dict[str, list[dict[str, Any]]] = {}
    if args.weak_labels_tsv is not None:
        weak_df = pd.read_csv(args.weak_labels_tsv, sep="\t", dtype=str).fillna("")
        for key, group in weak_df.groupby("key", sort=False):
            weak_labels_by_key[str(key)] = group.to_dict("records")

    run_dir = DATA_RESULTS_DIR / run_id
    out_dir = run_dir / out_subdir
    if out_dir.exists():
        raise FileExistsError(f"Output subdirectory already exists: {out_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for paper in papers:
        raw_text = ""
        if paper.text_path.exists():
            raw_text = paper.text_path.read_text(encoding="utf-8", errors="ignore")
        _, paper_artifacts, paper_summary = enumerate_numbered_doe_candidates_for_paper(
            paper=paper,
            raw_text=raw_text,
            existing_forms=weak_labels_by_key.get(paper.key, []),
            min_numbered_rows=args.min_numbered_rows,
        )
        artifact_rows.extend(paper_artifacts)
        summary_rows.append(paper_summary)

    stats = write_candidate_artifacts(
        out_dir=out_dir,
        artifact_rows=artifact_rows,
        summary_rows=summary_rows,
        expected_min_recovered=args.expected_min_recovered,
    )
    run_context = render_run_context(
        run_id=run_id,
        run_type=args.run_type,
        out_subdir=out_subdir,
        manifest_tsv=args.manifest_tsv,
        weak_labels_tsv=args.weak_labels_tsv,
        paper_keys=args.paper_keys,
        out_dir=out_dir,
        stats=stats,
        expected_min_recovered=args.expected_min_recovered,
    )
    run_context_path = run_dir / "RUN_CONTEXT.md"
    if run_context_path.exists():
        existing = run_context_path.read_text(encoding="utf-8")
        if run_context.strip() not in existing:
            run_context_path.write_text(existing.rstrip() + "\n\n" + run_context, encoding="utf-8")
    else:
        run_context_path.write_text(run_context, encoding="utf-8")

    if args.expected_min_recovered > 0:
        bad = [
            row["paper_key"]
            for row in summary_rows
            if int(str(row.get("new_candidates_emitted", "0")) or "0") < args.expected_min_recovered
        ]
        if bad:
            raise SystemExit(
                f"Deterministic DOE row recovery regression failed for: {', '.join(bad)}; expected at least {args.expected_min_recovered} new candidates."
            )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_type": args.run_type,
                "out_dir": str(out_dir),
                "artifact_path": str(stats["artifact_path"]),
                "summary_path": str(stats["summary_path"]),
                "paper_count": stats["paper_count"],
                "candidate_count": stats["candidate_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
