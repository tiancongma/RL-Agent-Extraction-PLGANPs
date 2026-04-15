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
- It does not perform full DOE coded-level decoding.
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


def parse_formulation_number(cell_text: str) -> int | None:
    match = re.fullmatch(r"(\d{1,3})\s*\.?", normalize_text(cell_text))
    if not match:
        return None
    return int(match.group(1))


def row_is_numbered(row: list[str]) -> int | None:
    if not row:
        return None
    return parse_formulation_number(row[0])


def count_numeric_like_cells(row: list[str]) -> int:
    count = 0
    for cell in row:
        if re.search(r"\d", cell):
            count += 1
    return count


def first_numbered_row_index(rows: list[list[str]]) -> int | None:
    for idx, row in enumerate(rows):
        number = row_is_numbered(row)
        if number is None:
            continue
        if count_numeric_like_cells(row[1:]) >= 3:
            return idx
    return None


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
        numbered_idx = first_numbered_row_index(rows)
        if numbered_idx is None:
            continue
        numbered_rows: list[list[str]] = []
        for row in rows[numbered_idx:]:
            if row_is_numbered(row) is None:
                continue
            if count_numeric_like_cells(row[1:]) < 3:
                continue
            numbered_rows.append(row)
        if len(numbered_rows) < min_numbered_rows:
            continue
        header_row = combine_header_rows(rows, numbered_idx)
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
    numbered_idx = first_numbered_row_index(rows)
    if numbered_idx is None:
        return None
    numbered_rows: list[list[str]] = []
    for row in rows[numbered_idx:]:
        if row_is_numbered(row) is None:
            continue
        if count_numeric_like_cells(row[1:]) < 3:
            continue
        numbered_rows.append(row)
    if len(numbered_rows) < min_numbered_rows:
        return None
    header_row = combine_header_rows(rows, numbered_idx)
    keyword_score = table_keyword_score(header_row, rows[:numbered_idx])
    if keyword_score < 2 and source_type != "semantic_authorized_table_target":
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
        if header_matches(clean_header, r"\bentrapment\b", r"\bencapsulation\b"):
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


def build_stage2_candidate_form(
    *,
    paper: PaperRecord,
    table_id: str,
    csv_path: Path,
    formulation_number: int,
    row: list[str],
    header_row: list[str],
    raw_text: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    row_text = " | ".join(cell for cell in row if cell)
    fields, extras = parse_row_fields(header_row=header_row, row=row, title=paper.title, raw_text=raw_text)
    change_descriptions = [f"{key}={value}" for key, value in sorted(extras.items()) if key not in {"polymer_identity", "polymer_name_raw"}]
    candidate = {
        "formulation_id": f"{paper.key}_DOE_Row_{formulation_number:02d}",
        "raw_formulation_label": f"{formulation_number}.",
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
    for idx, table in enumerate(selected_tables, start=1):
        table_id = normalize_text(table.get("semantic_table_id")) or f"{paper.key}__numbered_doe_table_{idx:02d}"
        selected_table_ids.append(table_id)
        selected_table_paths.append(str(Path(table["csv_path"])).replace("\\", "/"))
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

    notes = (
        "Explicit numbered DOE table rows were enumerated deterministically from authorized semantic table targets."
        if emitted_forms
        else "No numbered DOE rows were emitted from the authorized semantic table targets."
    )
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
