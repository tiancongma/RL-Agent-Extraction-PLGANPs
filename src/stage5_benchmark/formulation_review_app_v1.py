#!/usr/bin/env python3
from __future__ import annotations

"""
Serve a local formulation-centered review UI for frozen Stage5 audit artifacts.

Purpose:
- review one formulation object at a time
- keep boundary, value, and evidence decisions in an append-only review ledger
- provide a browser surface without adding a new runtime dependency

Inputs:
- explicit Stage5 final or audit-ready TSV
- optional Layer3 field review seed TSV

Outputs:
- formulation_review_decisions_v1.jsonl
- formulation_review_session_metadata_v1.json

Stage role:
- supporting Stage5 / Layer3 human review surface
- not a benchmark-governing final-output path
"""

import argparse
import csv
import json
import mimetypes
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


DECISIONS_NAME = "formulation_review_decisions_v1.jsonl"
METADATA_NAME = "formulation_review_session_metadata_v1.json"

BOUNDARY_DECISIONS = [
    "accept_as_gt_formulation",
    "blank_or_control_keep",
    "merge_with",
    "split_needed",
    "missing_formulation_from_source",
    "not_formulation",
    "commercial_comparator_exclude",
    "unclear_needs_second_review",
]

FIELD_DECISIONS = [
    "accept",
    "correct",
    "blank_not_reported",
    "unsupported",
    "unresolved_table",
    "normalization_pending",
    "unclear",
]

EVIDENCE_DECISIONS = [
    "supported",
    "unsupported_text",
    "unresolved_table",
    "normalization_pending",
    "no_field_evidence",
    "unclear",
]

WIDE_FIELD_SUFFIXES = ("_value", "_candidate")
STUDIED_VARIABLES_FIELD = "studied_variables_json"


@dataclass(frozen=True)
class ReviewSources:
    formulation_tsv: Path
    seed_rows_tsv: Path | None = None
    source_index_tsv: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--audit-ready-tsv",
        type=Path,
        help="Explicit final_formulation_table_audit_ready_v1.tsv path.",
    )
    source.add_argument(
        "--final-table-tsv",
        type=Path,
        help="Explicit Stage5 final_formulation_table_v1.tsv path.",
    )
    parser.add_argument(
        "--seed-rows-tsv",
        type=Path,
        default=None,
        help="Optional Layer3 field review seed TSV from workbook builder.",
    )
    parser.add_argument(
        "--source-index-tsv",
        type=Path,
        default=None,
        help="Optional explicit paper_key -> doi/pdf_path/html_path allowlist TSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for append-only reviewer decisions and session metadata.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of formulation cards to load for smoke tests.",
    )
    parser.add_argument(
        "--export-snapshot",
        action="store_true",
        help="Load inputs, write metadata, print a card-count snapshot, and exit.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input TSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Input TSV has no header: {path}")
        return [{key: value or "" for key, value in row.items()} for row in reader]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def first_nonblank(row: dict[str, str], names: list[str], default: str = "") -> str:
    for name in names:
        value = row.get(name, "")
        if value and value.strip():
            return value.strip()
    return default


def compact_dict(row: dict[str, str], names: list[str]) -> dict[str, str]:
    return {name: row.get(name, "") for name in names if row.get(name, "")}


def stable_formulation_id(row: dict[str, str], fallback_index: int) -> str:
    value = first_nonblank(
        row,
        [
            "formulation_id",
            "final_formulation_id",
            "source_formulation_id",
            "canonical_formulation_id",
            "stage5_formulation_id",
            "model_formulation_id",
            "original_model_formulation_id",
        ],
    )
    if value:
        return value
    label = first_nonblank(
        row,
        [
            "article_formulation_id",
            "article_formulation_label",
            "formulation_label_stage5",
            "formulation_label_params",
            "raw_formulation_label",
            "row_identity_description",
        ],
    )
    return label or f"row_{fallback_index + 1}"


def stable_paper_key(row: dict[str, str]) -> str:
    return first_nonblank(row, ["paper_key", "zotero_key", "key", "source_key"], "UNKNOWN")


def resolve_index_path(index_path: Path, value: str) -> Path | None:
    value = value.strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = index_path.parent / path
    return path.resolve()


def source_path_from_copied_files(index_path: Path, row: dict[str, str], kind: str) -> Path | None:
    raw = first_nonblank(row, ["copied_files_json"])
    if not raw:
        return None
    try:
        copied_files = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(copied_files, list):
        return None
    for item in copied_files:
        if not isinstance(item, dict):
            continue
        if str(item.get("kind", "")).strip() == kind:
            return resolve_index_path(index_path, str(item.get("path", "")))
    return None


def load_source_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = read_tsv(path)
    source_index: dict[str, dict[str, Any]] = {}
    for row in rows:
        paper_key = stable_paper_key(row)
        if not paper_key or paper_key == "UNKNOWN":
            continue
        pdf_path = resolve_index_path(path, first_nonblank(row, ["pdf_path", "source_pdf_path", "pdf"]))
        html_path = resolve_index_path(path, first_nonblank(row, ["html_path", "source_html_path", "html"]))
        if pdf_path is None:
            pdf_path = source_path_from_copied_files(path, row, "pdf")
        if html_path is None:
            html_path = source_path_from_copied_files(path, row, "html")
        source_index[paper_key] = {
            "paper_key": paper_key,
            "doi": first_nonblank(row, ["doi", "DOI"]),
            "pdf_path": pdf_path,
            "html_path": html_path,
        }
    return source_index


def public_source_controls(paper_key: str, doi: str, source_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = source_index.get(paper_key, {})
    indexed_doi = str(source.get("doi") or "").strip()
    pdf_path = source.get("pdf_path")
    html_path = source.get("html_path")
    return {
        "doi": indexed_doi or doi,
        "pdf_available": bool(pdf_path and Path(pdf_path).exists()),
        "html_available": bool(html_path and Path(html_path).exists()),
    }


def build_field_rows(seed_rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for index, row in enumerate(seed_rows):
        paper_key = stable_paper_key(row)
        formulation_id = stable_formulation_id(row, index)
        field = {
            "paper_key": paper_key,
            "formulation_id": formulation_id,
            "field_name": first_nonblank(row, ["field_name", "parameter", "value_field"]),
            "extracted_value": first_nonblank(row, ["extracted_value", "system_value", "value"]),
            "extracted_unit": first_nonblank(row, ["extracted_unit", "system_unit", "unit"]),
            "evidence_text": first_nonblank(row, ["evidence_text", "field_evidence_text", "evidence"]),
            "evidence_anchor_text": first_nonblank(row, ["evidence_anchor_text", "anchor_text"]),
            "evidence_source_type": first_nonblank(row, ["evidence_source_type", "source_type"]),
            "evidence_status_detail": first_nonblank(row, ["evidence_status_detail", "evidence_status"]),
            "review_warning": first_nonblank(row, ["review_warning", "warning"]),
            "normalization_status": first_nonblank(row, ["normalization_status"]),
            "gt_status": first_nonblank(row, ["gt_status"]),
            "gt_value": first_nonblank(row, ["gt_value"]),
            "gt_unit": first_nonblank(row, ["gt_unit"]),
            "notes": first_nonblank(row, ["notes", "review_notes"]),
        }
        grouped.setdefault((paper_key, formulation_id), []).append(field)
    return grouped


def normalize_variable_name(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def studied_variable_family(name: Any, unit: Any = "") -> str:
    clean_name = normalize_variable_name(name)
    clean_unit = normalize_variable_name(unit)
    if re.search(r"\bhomogeni[sz](?:ation|er)?_?speed\b", clean_name):
        return "homogenization_speed_rpm"
    if clean_unit and clean_unit not in clean_name:
        return f"{clean_name}_{clean_unit}".strip("_")
    return clean_name


def split_studied_variable_value_unit(value: Any, fallback_unit: str = "") -> tuple[str, str]:
    text = normalize_text(value)
    fallback_unit = normalize_text(fallback_unit)
    match = re.search(
        r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(rpm|mg\s*/\s*mL|mg/ml|%\s*w\s*/\s*v|%w/v|%|mL|ml|mg|h|hr|hours?|min|minutes?)\b",
        text,
        flags=re.I,
    )
    if match:
        return match.group(1).replace(",", ""), normalize_text(match.group(2))
    if fallback_unit and re.fullmatch(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text):
        return text.replace(",", ""), fallback_unit
    return text, fallback_unit


def studied_variables_from_json(row: dict[str, str], paper_key: str, formulation_id: str) -> list[dict[str, str]]:
    raw = row.get(STUDIED_VARIABLES_FIELD, "")
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    fields: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        family = first_nonblank(item, ["variable_family"]) or studied_variable_family(item.get("variable_name"), item.get("unit"))
        value, unit = split_studied_variable_value_unit(item.get("value", ""), item.get("unit", ""))
        if not family or not value:
            continue
        fields.append(
            {
                "paper_key": paper_key,
                "formulation_id": formulation_id,
                "field_name": family,
                "extracted_value": value,
                "extracted_unit": unit,
                "evidence_text": normalize_text(item.get("evidence_text", "")),
                "evidence_anchor_text": "",
                "evidence_source_type": normalize_text(item.get("source", "")) or STUDIED_VARIABLES_FIELD,
                "evidence_status_detail": "studied_variable",
                "review_warning": "",
                "normalization_status": "",
                "gt_status": "",
                "gt_value": "",
                "gt_unit": "",
                "notes": normalize_text(item.get("scope", "")),
            }
        )
    return fields


def studied_variables_from_row_text(row: dict[str, str], paper_key: str, formulation_id: str) -> list[dict[str, str]]:
    text = normalize_text(
        " ".join(
            row.get(name, "")
            for name in (
                "row_identity_description",
                "raw_formulation_label",
                "evidence_row_identity",
                "evidence_composition",
            )
        )
    )
    if not text:
        return []
    fields: list[dict[str, str]] = []
    for match in re.finditer(
        r"\b(homogeni[sz](?:ation|er)?\s+speed)\s*(?:of|=|:)?\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(rpm)\b",
        text,
        flags=re.I,
    ):
        value = match.group(2).replace(",", "")
        unit = normalize_text(match.group(3))
        fields.append(
            {
                "paper_key": paper_key,
                "formulation_id": formulation_id,
                "field_name": studied_variable_family(match.group(1), unit),
                "extracted_value": value,
                "extracted_unit": unit,
                "evidence_text": text,
                "evidence_anchor_text": match.group(0),
                "evidence_source_type": "row_identity_description",
                "evidence_status_detail": "studied_variable",
                "review_warning": "",
                "normalization_status": "",
                "gt_status": "",
                "gt_value": "",
                "gt_unit": "",
                "notes": "formulation_row",
            }
        )
    return fields


def evidence_for_wide_field(row: dict[str, str], field_name: str) -> str:
    metric_terms = ("efficiency", "loading", "size", "pdi", "zeta", "dl_", "metric")
    composition_terms = ("polymer", "drug", "payload", "solvent", "stabilizer", "surfactant", "la_ga", "mw")
    preparation_terms = ("method", "stirring", "evaporation", "sonication", "homogenization", "centrifugation", "phase", "pH")
    if any(term in field_name for term in metric_terms):
        return first_nonblank(row, ["evidence_metrics", "evidence_row_identity"])
    if any(term in field_name for term in composition_terms):
        return first_nonblank(row, ["evidence_composition", "evidence_row_identity"])
    if any(term in field_name for term in preparation_terms):
        return first_nonblank(row, ["evidence_preparation", "evidence_row_identity"])
    return first_nonblank(row, ["evidence_row_identity", "evidence_metrics", "evidence_composition", "evidence_preparation"])


def wide_value_fields_from_row(row: dict[str, str], paper_key: str, formulation_id: str) -> list[dict[str, str]]:
    ignored = {
        "paper_level_formulation_status",
        "paper_level_ee_status",
        "paper_risk_level",
    }
    fields: list[dict[str, str]] = []
    fields.extend(studied_variables_from_json(row, paper_key, formulation_id))
    fields.extend(studied_variables_from_row_text(row, paper_key, formulation_id))
    for name, value in row.items():
        if name in ignored or not value or not value.strip():
            continue
        if not name.endswith(WIDE_FIELD_SUFFIXES):
            continue
        field_name = name
        extracted_unit = ""
        if name.endswith("_value"):
            field_name = name[: -len("_value")]
            unit_value = row.get(f"{field_name}_unit", "")
            if unit_value:
                extracted_unit = unit_value
        fields.append(
            {
                "paper_key": paper_key,
                "formulation_id": formulation_id,
                "field_name": field_name,
                "extracted_value": value.strip(),
                "extracted_unit": extracted_unit.strip(),
                "evidence_text": evidence_for_wide_field(row, field_name),
                "evidence_anchor_text": "",
                "evidence_source_type": "wide_final_table",
                "evidence_status_detail": "",
                "review_warning": "",
                "normalization_status": first_nonblank(row, ["normalization_review_reasons"]),
                "gt_status": "",
                "gt_value": "",
                "gt_unit": "",
                "notes": "",
            }
        )
    dedup: dict[tuple[str, str, str], dict[str, str]] = {}
    for field in fields:
        key = (field["field_name"], field["extracted_value"], field["extracted_unit"])
        if key not in dedup:
            dedup[key] = field
    return list(dedup.values())


def build_review_cards(
    formulation_rows: list[dict[str, str]],
    seed_rows: list[dict[str, str]] | None = None,
    source_index: dict[str, dict[str, Any]] | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
    field_rows = build_field_rows(seed_rows or [])
    source_index = source_index or {}
    cards_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    inferred_field_rows: dict[tuple[str, str], list[dict[str, str]]] = {}

    for index, row in enumerate(formulation_rows):
        paper_key = stable_paper_key(row)
        formulation_id = stable_formulation_id(row, index)
        key = (paper_key, formulation_id)
        if key not in cards_by_key:
            doi = first_nonblank(row, ["doi", "DOI"])
            cards_by_key[key] = {
                "paper_key": paper_key,
                "formulation_id": formulation_id,
                "paper_title": first_nonblank(row, ["paper_title", "title", "article_title"]),
                "doi": doi,
                "source_controls": public_source_controls(paper_key, doi, source_index),
                "article_formulation_id": first_nonblank(row, ["article_formulation_id"]),
                "article_formulation_label": first_nonblank(row, ["article_formulation_label", "raw_formulation_label"]),
                "formulation_label_stage5": first_nonblank(row, ["formulation_label_stage5", "model_formulation_id"]),
                "formulation_label_params": first_nonblank(row, ["formulation_label_params", "row_identity_description"]),
                "variant_role": first_nonblank(row, ["variant_role", "row_role", "formulation_role"]),
                "payload_state": first_nonblank(row, ["payload_state", "closure_status", "decision_status"]),
                "risk": first_nonblank(row, ["paper_risk_level", "risk_level", "review_priority"]),
                "evidence_text": first_nonblank(
                    row,
                    [
                        "evidence_text",
                        "representative_evidence_text",
                        "source_evidence_text",
                        "supporting_evidence_text",
                        "evidence_row_identity",
                        "evidence_metrics",
                        "evidence_composition",
                        "evidence_preparation",
                    ],
                ),
                "source_summary": compact_dict(
                    row,
                    [
                        "source_table_reference",
                        "source_section",
                        "source_locator",
                        "supporting_evidence_refs",
                        "final_output_decision_rule",
                    ],
                ),
                "identity": compact_dict(
                    row,
                    [
                        "formulation_core_signature",
                        "identity_signature",
                        "article_native_label",
                        "formulation_family_id",
                    ],
                ),
                "fields": [],
                "source_row_count": 0,
            }
        cards_by_key[key]["source_row_count"] += 1
        if key not in field_rows:
            inferred_field_rows.setdefault(key, []).extend(wide_value_fields_from_row(row, paper_key, formulation_id))

    for key, fields in field_rows.items():
        if key in cards_by_key:
            cards_by_key[key]["fields"] = fields
    for key, fields in inferred_field_rows.items():
        if key in cards_by_key and not cards_by_key[key]["fields"]:
            cards_by_key[key]["fields"] = fields

    cards = list(cards_by_key.values())
    cards.sort(key=lambda item: (item["paper_key"], item["formulation_id"]))
    if limit > 0:
        cards = cards[:limit]
    return cards


def write_metadata(
    out_dir: Path,
    session_id: str,
    sources: ReviewSources,
    card_count: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "review_session_id": session_id,
        "created_at": utc_now(),
        "stage_role": "supporting_nondefault_human_review_surface",
        "benchmark_valid": False,
        "input_formulation_tsv": str(sources.formulation_tsv),
        "input_seed_rows_tsv": str(sources.seed_rows_tsv) if sources.seed_rows_tsv else "",
        "input_source_index_tsv": str(sources.source_index_tsv) if sources.source_index_tsv else "",
        "decision_jsonl": str(out_dir / DECISIONS_NAME),
        "card_count": card_count,
        "boundary_decision_options": BOUNDARY_DECISIONS,
        "field_decision_options": FIELD_DECISIONS,
        "evidence_decision_options": EVIDENCE_DECISIONS,
    }
    metadata_path = out_dir / METADATA_NAME
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def append_decision(
    out_dir: Path,
    session_id: str,
    sources: ReviewSources,
    decision: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "review_session_id": session_id,
        "reviewed_at": utc_now(),
        "input_formulation_tsv": str(sources.formulation_tsv),
        "input_seed_rows_tsv": str(sources.seed_rows_tsv) if sources.seed_rows_tsv else "",
        "input_source_index_tsv": str(sources.source_index_tsv) if sources.source_index_tsv else "",
        "paper_key": str(decision.get("paper_key", "")).strip(),
        "formulation_id": str(decision.get("formulation_id", "")).strip(),
        "decision_layer": str(decision.get("decision_layer", "")).strip(),
        "decision": str(decision.get("decision", "")).strip(),
        "target_formulation_id": str(decision.get("target_formulation_id", "")).strip(),
        "field_name": str(decision.get("field_name", "")).strip(),
        "gt_value": str(decision.get("gt_value", "")).strip(),
        "gt_unit": str(decision.get("gt_unit", "")).strip(),
        "evidence_status_override": str(decision.get("evidence_status_override", "")).strip(),
        "reviewer_note": str(decision.get("reviewer_note", "")).strip(),
    }
    if not record["paper_key"] or not record["formulation_id"]:
        raise ValueError("Decision requires paper_key and formulation_id")
    if record["decision_layer"] not in {"boundary", "field", "evidence"}:
        raise ValueError("decision_layer must be boundary, field, or evidence")
    if not record["decision"]:
        raise ValueError("Decision requires a nonblank decision value")

    decision_path = out_dir / DECISIONS_NAME
    with decision_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def read_decisions(out_dir: Path) -> list[dict[str, Any]]:
    decision_path = out_dir / DECISIONS_NAME
    if not decision_path.exists():
        return []
    rows = []
    with decision_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ReviewServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        cards: list[dict[str, Any]],
        out_dir: Path,
        session_id: str,
        sources: ReviewSources,
        source_index: dict[str, dict[str, Any]],
    ) -> None:
        super().__init__(server_address, handler_class)
        self.cards = cards
        self.out_dir = out_dir
        self.session_id = session_id
        self.sources = sources
        self.source_index = source_index
        self.started_at = time.time()


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("[formulation-review-ui] " + format % args + "\n")

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self) -> None:
        body = HTML_APP.encode("utf-8")
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_source_file(self, paper_key: str, kind: str, *, head_only: bool = False) -> None:
        if kind not in {"pdf", "html"}:
            self.send_error(HTTPStatus.NOT_FOUND.value, "Unknown source kind")
            return
        source = self.server.source_index.get(paper_key)
        if not source:
            self.send_error(HTTPStatus.NOT_FOUND.value, "Paper key is not in source allowlist")
            return
        path = source.get(f"{kind}_path")
        if not path:
            self.send_error(HTTPStatus.NOT_FOUND.value, "Source file is not registered")
            return
        path = Path(path)
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND.value, "Registered source file does not exist")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = b"" if head_only else path.read_bytes()
        content_length = path.stat().st_size if head_only else len(body)
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html()
            return
        if parsed.path.startswith("/source/"):
            parts = parsed.path.split("/")
            if len(parts) != 4:
                self.send_error(HTTPStatus.NOT_FOUND.value, "Invalid source route")
                return
            paper_key = unquote(parts[2])
            kind = unquote(parts[3])
            self.send_source_file(paper_key, kind)
            return
        if parsed.path == "/api/health":
            self.send_json(
                {
                    "ok": True,
                    "review_session_id": self.server.session_id,
                    "card_count": len(self.server.cards),
                    "decision_count": len(read_decisions(self.server.out_dir)),
                    "input_formulation_tsv": str(self.server.sources.formulation_tsv),
                    "input_seed_rows_tsv": str(self.server.sources.seed_rows_tsv or ""),
                    "input_source_index_tsv": str(self.server.sources.source_index_tsv or ""),
                    "source_index_paper_count": len(self.server.source_index),
                }
            )
            return
        if parsed.path == "/api/cards":
            params = parse_qs(parsed.query)
            q = (params.get("q", [""])[0] or "").strip().lower()
            cards = self.server.cards
            if q:
                cards = [
                    card
                    for card in cards
                    if q in card.get("paper_key", "").lower()
                    or q in card.get("formulation_id", "").lower()
                    or q in card.get("paper_title", "").lower()
                ]
            self.send_json(
                {
                    "cards": cards,
                    "boundary_decision_options": BOUNDARY_DECISIONS,
                    "field_decision_options": FIELD_DECISIONS,
                    "evidence_decision_options": EVIDENCE_DECISIONS,
                }
            )
            return
        if parsed.path == "/api/decisions":
            self.send_json({"decisions": read_decisions(self.server.out_dir)})
            return
        self.send_error(HTTPStatus.NOT_FOUND.value, "Not found")

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/source/"):
            parts = parsed.path.split("/")
            if len(parts) != 4:
                self.send_error(HTTPStatus.NOT_FOUND.value, "Invalid source route")
                return
            paper_key = unquote(parts[2])
            kind = unquote(parts[3])
            self.send_source_file(paper_key, kind, head_only=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND.value, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/decision":
            self.send_error(HTTPStatus.NOT_FOUND.value, "Not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw_body or "{}")
            record = append_decision(
                self.server.out_dir,
                self.server.session_id,
                self.server.sources,
                payload,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as API validation error
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self.send_json({"ok": True, "record": record})


HTML_APP = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Formulation Review</title>
  <style>
    :root {
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #20242c;
      --muted: #687083;
      --line: #d8deea;
      --accent: #0f766e;
      --accent-ink: #ffffff;
      --warn: #9a3412;
      --shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      letter-spacing: 0;
    }
    button, input, select, textarea { font: inherit; }
    .shell {
      display: grid;
      grid-template-columns: minmax(230px, 300px) minmax(360px, 1fr) minmax(320px, 420px);
      min-height: 100vh;
      align-items: start;
    }
    .sidebar, .main, .detail {
      min-width: 0;
      padding: 16px;
    }
    .sidebar {
      border-right: 1px solid var(--line);
      background: #eef3f2;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: hidden;
    }
    .detail {
      border-left: 1px solid var(--line);
      background: #f9faf6;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    h1, h2, h3 { margin: 0; line-height: 1.2; }
    h1 { font-size: 18px; }
    h2 { font-size: 16px; margin-bottom: 10px; }
    h3 { font-size: 14px; margin-bottom: 8px; }
    .topline {
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }
    .badge {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      color: var(--muted);
      background: rgba(255,255,255,0.75);
      font-size: 12px;
      white-space: nowrap;
    }
    .search {
      width: 100%;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 9px;
      margin-bottom: 12px;
      background: white;
    }
    .nav-stack {
      display: grid;
      grid-template-rows: minmax(150px, 0.44fr) minmax(220px, 0.56fr);
      gap: 12px;
      height: calc(100vh - 108px);
      min-height: 0;
    }
    .nav-section {
      min-height: 0;
      display: flex;
      flex-direction: column;
    }
    .nav-heading {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .list {
      display: grid;
      align-content: start;
      gap: 8px;
      min-height: 0;
      overflow: auto;
      padding-right: 2px;
    }
    .list button {
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      padding: 9px;
      cursor: pointer;
      min-height: 72px;
      box-shadow: var(--shadow);
    }
    .list button.active { border-color: var(--accent); outline: 2px solid rgba(15, 118, 110, 0.12); }
    .paper-list button {
      min-height: 58px;
    }
    .paper-list .label {
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .paper-count {
      float: right;
      color: var(--muted);
      font-weight: 600;
    }
    .key { font-size: 12px; color: var(--muted); overflow-wrap: anywhere; }
    .label { font-size: 14px; font-weight: 650; margin-top: 3px; overflow-wrap: anywhere; }
    .sub { font-size: 12px; color: var(--muted); margin-top: 3px; overflow-wrap: anywhere; }
    .section {
      border-top: 1px solid var(--line);
      padding-top: 14px;
      margin-top: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: var(--shadow);
    }
    .paper-title {
      color: var(--muted);
      font-size: 13px;
      margin-top: 5px;
      overflow-wrap: anywhere;
    }
    .kv {
      display: grid;
      grid-template-columns: minmax(100px, 150px) minmax(0, 1fr);
      gap: 7px 10px;
      font-size: 13px;
    }
    .kv dt { color: var(--muted); }
    .kv dd { margin: 0; overflow-wrap: anywhere; }
    .decision-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 8px;
      margin-top: 12px;
    }
    select, input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      padding: 8px;
      min-height: 36px;
    }
    textarea { min-height: 80px; resize: vertical; }
    .full { grid-column: 1 / -1; }
    .primary {
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: var(--accent-ink);
      min-height: 38px;
      padding: 8px 12px;
      cursor: pointer;
    }
    .field-list {
      display: grid;
      gap: 8px;
    }
    .field-row {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      padding: 10px;
      cursor: pointer;
    }
    .field-row.active { border-color: var(--accent); }
    .field-name { font-weight: 700; font-size: 13px; }
    .value { margin-top: 3px; font-size: 13px; overflow-wrap: anywhere; }
    .warning { color: var(--warn); font-size: 12px; margin-top: 3px; overflow-wrap: anywhere; }
    .saved-review {
      border: 1px solid #b7d7cf;
      border-radius: 6px;
      background: #f2faf7;
      color: #143f3a;
      font-size: 13px;
      margin-top: 12px;
      padding: 10px;
      overflow-wrap: anywhere;
    }
    .evidence {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: #293241;
      background: #fbfcfe;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      font-size: 13px;
      max-height: 260px;
      overflow: auto;
    }
    .status {
      min-height: 24px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .source-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .source-actions a, .source-actions span {
      min-height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 9px;
      background: white;
      color: var(--ink);
      text-decoration: none;
      font-size: 13px;
    }
    .source-actions span { color: var(--muted); }
    #evidencePanel {
      flex: 1;
      min-height: 0;
      overflow: auto;
    }
    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; }
      .sidebar, .detail {
        border: 0;
        position: static;
        height: auto;
        overflow: visible;
      }
      .nav-stack {
        height: auto;
        grid-template-rows: auto;
      }
      .list, .field-list { max-height: none; }
      #evidencePanel { overflow: visible; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="topline">
        <h1>Formulation Review</h1>
        <span class="badge" id="countBadge">0</span>
      </div>
      <input class="search" id="search" type="search" placeholder="Filter paper or formulation">
      <div class="nav-stack">
        <section class="nav-section">
          <div class="nav-heading">
            <span>Papers</span>
            <span id="paperBadge">0</span>
          </div>
          <div class="list paper-list" id="paperList"></div>
        </section>
        <section class="nav-section">
          <div class="nav-heading">
            <span>Formulations</span>
            <span id="formulationBadge">0</span>
          </div>
          <div class="list" id="cardList"></div>
        </section>
      </div>
    </aside>
    <main class="main">
      <div class="panel" id="cardPanel"></div>
      <section class="section">
        <h2>Boundary Review</h2>
        <div class="panel">
          <div class="decision-grid">
            <select id="boundaryDecision"></select>
            <input id="targetFormulation" placeholder="Merge/split target id">
            <textarea class="full" id="boundaryNote" placeholder="Reviewer note"></textarea>
            <button class="primary full" id="saveBoundary" title="Record boundary decision">Save boundary decision</button>
          </div>
          <div class="status" id="boundaryStatus"></div>
        </div>
      </section>
      <section class="section">
        <h2>Value Review</h2>
        <div class="field-list" id="fieldList"></div>
      </section>
    </main>
    <aside class="detail">
      <h2>Evidence Review</h2>
      <div class="panel" id="evidencePanel"></div>
    </aside>
  </div>
  <script>
    let cards = [];
    let options = {};
    let decisions = [];
    let selectedCard = null;
    let selectedField = null;
    let selectedPaperKey = "";

    const $ = (id) => document.getElementById(id);

    function optionNodes(values) {
      return values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("");
    }

    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[char]));
    }

    function decisionKey(layer, card, fieldName) {
      if (!card) return "";
      return [layer, card.paper_key || "", card.formulation_id || "", fieldName || ""].join("||");
    }

    function latestDecisionMap() {
      const latest = {};
      for (const decision of decisions) {
        const key = [
          decision.decision_layer || "",
          decision.paper_key || "",
          decision.formulation_id || "",
          decision.field_name || ""
        ].join("||");
        latest[key] = decision;
      }
      return latest;
    }

    function latestDecision(layer, card, fieldName) {
      return latestDecisionMap()[decisionKey(layer, card, fieldName)] || null;
    }

    function setSelectValue(id, value) {
      const element = $(id);
      if (!element || !value) return;
      const hasOption = Array.from(element.options).some((option) => option.value === value);
      if (hasOption) element.value = value;
    }

    function savedReviewSummary(record) {
      if (!record) return "";
      const parts = [
        record.decision ? `decision: ${record.decision}` : "",
        record.evidence_status_override ? `evidence: ${record.evidence_status_override}` : "",
        [record.gt_value, record.gt_unit].filter(Boolean).join(" ") ? `GT: ${[record.gt_value, record.gt_unit].filter(Boolean).join(" ")}` : "",
        record.reviewer_note ? `note: ${record.reviewer_note}` : ""
      ].filter(Boolean);
      return `
        <div class="saved-review">
          <strong>Saved review</strong>
          <div>${escapeHtml(parts.join(" | "))}</div>
          <div class="status">${escapeHtml(record.reviewed_at || "")}</div>
        </div>
      `;
    }

    function cardSearchText(card) {
      return [
        card.paper_key,
        card.formulation_id,
        card.paper_title,
        card.article_formulation_label,
        card.formulation_label_stage5,
        card.formulation_label_params,
        card.row_identity_description
      ].join(" ").toLowerCase();
    }

    function groupCardsByPaper() {
      const groups = [];
      const index = {};
      for (const card of cards) {
        const paperKey = card.paper_key || "UNKNOWN";
        if (!index[paperKey]) {
          index[paperKey] = {
            paper_key: paperKey,
            doi: card.doi || "",
            title: card.paper_title || "",
            cards: []
          };
          groups.push(index[paperKey]);
        }
        index[paperKey].cards.push(card);
        if (!index[paperKey].title && card.paper_title) index[paperKey].title = card.paper_title;
        if (!index[paperKey].doi && card.doi) index[paperKey].doi = card.doi;
      }
      return groups;
    }

    function visibleCardsForSelectedPaper() {
      if (!selectedPaperKey) return [];
      return cards.filter((card) => (card.paper_key || "UNKNOWN") === selectedPaperKey);
    }

    async function loadCards() {
      const query = $("search").value.trim();
      const [cardsResponse, decisionsResponse] = await Promise.all([
        fetch(`/api/cards${query ? `?q=${encodeURIComponent(query)}` : ""}`),
        fetch("/api/decisions")
      ]);
      const payload = await cardsResponse.json();
      const decisionsPayload = await decisionsResponse.json();
      cards = payload.cards || [];
      decisions = decisionsPayload.decisions || [];
      options = payload;
      $("boundaryDecision").innerHTML = optionNodes(options.boundary_decision_options || []);
      const paperGroups = groupCardsByPaper();
      $("countBadge").textContent = `${paperGroups.length} papers | ${cards.length} cards`;
      if (!paperGroups.some((group) => group.paper_key === selectedPaperKey)) {
        selectedPaperKey = paperGroups.length ? paperGroups[0].paper_key : "";
      }
      const paperCards = visibleCardsForSelectedPaper();
      if (!selectedCard || !paperCards.includes(selectedCard)) {
        selectedCard = paperCards[0] || cards[0] || null;
      }
      selectedPaperKey = selectedCard ? selectedCard.paper_key : selectedPaperKey;
      selectedField = selectedCard && selectedCard.fields.length ? selectedCard.fields[0] : null;
      render();
    }

    function render() {
      renderPaperList();
      renderFormulationList();
      renderCard();
      renderFields();
      renderEvidence();
    }

    function renderPaperList() {
      const paperGroups = groupCardsByPaper();
      $("paperBadge").textContent = `${paperGroups.length}`;
      $("paperList").innerHTML = paperGroups.map((group, index) => `
        <button class="${selectedPaperKey === group.paper_key ? "active" : ""}" data-index="${index}">
          <div class="key">${escapeHtml(group.paper_key)} <span class="paper-count">${group.cards.length}</span></div>
          <div class="label">${escapeHtml(group.title || group.paper_key)}</div>
          <div class="sub">${escapeHtml(group.doi || "DOI not available")}</div>
        </button>
      `).join("");
      for (const button of $("paperList").querySelectorAll("button")) {
        button.addEventListener("click", () => {
          const group = paperGroups[Number(button.dataset.index)];
          selectedPaperKey = group.paper_key;
          selectedCard = group.cards[0] || null;
          selectedField = selectedCard && selectedCard.fields.length ? selectedCard.fields[0] : null;
          render();
        });
      }
    }

    function renderFormulationList() {
      const paperCards = visibleCardsForSelectedPaper();
      $("formulationBadge").textContent = `${paperCards.length}`;
      $("cardList").innerHTML = paperCards.map((card, index) => `
        <button class="${selectedCard === card ? "active" : ""}" data-index="${index}">
          <div class="key">${escapeHtml(card.paper_key)}</div>
          <div class="label">${escapeHtml(card.formulation_id)}</div>
          <div class="sub">${escapeHtml(card.article_formulation_label || card.formulation_label_stage5 || card.formulation_label_params || "unlabeled")}</div>
        </button>
      `).join("");
      for (const button of $("cardList").querySelectorAll("button")) {
        button.addEventListener("click", () => {
          selectedCard = paperCards[Number(button.dataset.index)];
          selectedPaperKey = selectedCard ? selectedCard.paper_key : selectedPaperKey;
          selectedField = selectedCard && selectedCard.fields.length ? selectedCard.fields[0] : null;
          render();
        });
      }
    }

    function renderCard() {
      if (!selectedCard) {
        $("cardPanel").innerHTML = "<h2>No formulation cards loaded</h2>";
        return;
      }
      const sourceControls = selectedCard.source_controls || {};
      const encodedKey = encodeURIComponent(selectedCard.paper_key);
      const sourceButtons = `
        <div class="source-actions">
          <span>DOI: ${escapeHtml(sourceControls.doi || selectedCard.doi || "not available")}</span>
          ${sourceControls.pdf_available ? `<a href="/source/${encodedKey}/pdf" target="_blank" rel="noopener">Open PDF</a>` : `<span>PDF not indexed</span>`}
          ${sourceControls.html_available ? `<a href="/source/${encodedKey}/html" target="_blank" rel="noopener">Open HTML</a>` : `<span>HTML not indexed</span>`}
        </div>
      `;
      const boundaryRecord = latestDecision("boundary", selectedCard, "");
      $("cardPanel").innerHTML = `
        <h2>${escapeHtml(selectedCard.formulation_id)}</h2>
        <div class="paper-title">${escapeHtml(selectedCard.paper_title || selectedCard.paper_key)}</div>
        ${sourceButtons}
        ${savedReviewSummary(boundaryRecord)}
        <div class="section">
          <dl class="kv">
            <dt>paper_key</dt><dd>${escapeHtml(selectedCard.paper_key)}</dd>
            <dt>doi</dt><dd>${escapeHtml(selectedCard.doi)}</dd>
            <dt>article label</dt><dd>${escapeHtml(selectedCard.article_formulation_label || selectedCard.article_formulation_id)}</dd>
            <dt>stage5 label</dt><dd>${escapeHtml(selectedCard.formulation_label_stage5 || selectedCard.formulation_label_params)}</dd>
            <dt>role</dt><dd>${escapeHtml(selectedCard.variant_role)}</dd>
            <dt>state</dt><dd>${escapeHtml(selectedCard.payload_state)}</dd>
            <dt>risk</dt><dd>${escapeHtml(selectedCard.risk)}</dd>
            <dt>source rows</dt><dd>${escapeHtml(selectedCard.source_row_count)}</dd>
          </dl>
        </div>
      `;
      if (boundaryRecord) {
        setSelectValue("boundaryDecision", boundaryRecord.decision);
        $("targetFormulation").value = boundaryRecord.target_formulation_id || "";
        $("boundaryNote").value = boundaryRecord.reviewer_note || "";
        $("boundaryStatus").textContent = `Saved ${boundaryRecord.decision} at ${boundaryRecord.reviewed_at}`;
      } else {
        $("targetFormulation").value = "";
        $("boundaryNote").value = "";
        $("boundaryStatus").textContent = "";
      }
    }

    function renderFields() {
      if (!selectedCard || !selectedCard.fields.length) {
        $("fieldList").innerHTML = `<div class="panel">No field seed rows attached for this formulation.</div>`;
        return;
      }
      $("fieldList").innerHTML = selectedCard.fields.map((field, index) => `
        <button class="field-row ${selectedField === field ? "active" : ""}" data-index="${index}">
          <div class="field-name">${escapeHtml(field.field_name || "unnamed_field")}</div>
          <div class="value">${escapeHtml([field.extracted_value, field.extracted_unit].filter(Boolean).join(" "))}</div>
          ${field.review_warning ? `<div class="warning">${escapeHtml(field.review_warning)}</div>` : ""}
        </button>
      `).join("");
      for (const row of $("fieldList").querySelectorAll("button")) {
        row.addEventListener("click", () => {
          selectedField = selectedCard.fields[Number(row.dataset.index)];
          render();
        });
      }
    }

    function renderEvidence() {
      if (!selectedCard) {
        $("evidencePanel").innerHTML = "";
        return;
      }
      const field = selectedField || {};
      const fieldRecord = selectedField ? latestDecision("field", selectedCard, selectedField.field_name) : null;
      const evidence = field.evidence_text || selectedCard.evidence_text || "";
      $("evidencePanel").innerHTML = `
        <h3>${escapeHtml(field.field_name || "Formulation evidence")}</h3>
        ${savedReviewSummary(fieldRecord)}
        <dl class="kv">
          <dt>value</dt><dd>${escapeHtml([field.extracted_value, field.extracted_unit].filter(Boolean).join(" "))}</dd>
          <dt>source type</dt><dd>${escapeHtml(field.evidence_source_type || "")}</dd>
          <dt>status</dt><dd>${escapeHtml(field.evidence_status_detail || "")}</dd>
          <dt>normalization</dt><dd>${escapeHtml(field.normalization_status || "")}</dd>
        </dl>
        <div class="section evidence">${escapeHtml(evidence || "No evidence text available.")}</div>
        <div class="section decision-grid">
          <select id="fieldDecision">${optionNodes(options.field_decision_options || [])}</select>
          <select id="evidenceDecision">${optionNodes(options.evidence_decision_options || [])}</select>
          <input id="gtValue" placeholder="GT value">
          <input id="gtUnit" placeholder="GT unit">
          <textarea class="full" id="fieldNote" placeholder="Reviewer note"></textarea>
          <button class="primary full" id="saveField" title="Record field and evidence decision">Save field decision</button>
        </div>
        <div class="status" id="fieldStatus"></div>
      `;
      if (fieldRecord) {
        setSelectValue("fieldDecision", fieldRecord.decision);
        setSelectValue("evidenceDecision", fieldRecord.evidence_status_override);
        $("gtValue").value = fieldRecord.gt_value || "";
        $("gtUnit").value = fieldRecord.gt_unit || "";
        $("fieldNote").value = fieldRecord.reviewer_note || "";
        $("fieldStatus").textContent = `Saved ${fieldRecord.field_name || "field"} at ${fieldRecord.reviewed_at}`;
      }
      $("saveField").addEventListener("click", saveField);
    }

    async function postDecision(payload) {
      const response = await fetch("/api/decision", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const result = await response.json();
      if (!result.ok) throw new Error(result.error || "save failed");
      return result.record;
    }

    async function saveBoundary() {
      if (!selectedCard) return;
      $("boundaryStatus").textContent = "Saving...";
      try {
        const record = await postDecision({
          paper_key: selectedCard.paper_key,
          formulation_id: selectedCard.formulation_id,
          decision_layer: "boundary",
          decision: $("boundaryDecision").value,
          target_formulation_id: $("targetFormulation").value,
          reviewer_note: $("boundaryNote").value
        });
        decisions.push(record);
        $("boundaryStatus").textContent = `Saved ${record.decision} at ${record.reviewed_at}`;
        renderCard();
      } catch (error) {
        $("boundaryStatus").textContent = error.message;
      }
    }

    async function saveField() {
      if (!selectedCard || !selectedField) return;
      $("fieldStatus").textContent = "Saving...";
      try {
        const record = await postDecision({
          paper_key: selectedCard.paper_key,
          formulation_id: selectedCard.formulation_id,
          decision_layer: "field",
          decision: $("fieldDecision").value,
          field_name: selectedField.field_name,
          gt_value: $("gtValue").value,
          gt_unit: $("gtUnit").value,
          evidence_status_override: $("evidenceDecision").value,
          reviewer_note: $("fieldNote").value
        });
        decisions.push(record);
        $("fieldStatus").textContent = `Saved ${record.field_name || "field"} at ${record.reviewed_at}`;
        renderFields();
        renderEvidence();
      } catch (error) {
        $("fieldStatus").textContent = error.message;
      }
    }

    $("search").addEventListener("input", () => loadCards());
    $("saveBoundary").addEventListener("click", saveBoundary);
    loadCards();
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    formulation_tsv = args.audit_ready_tsv or args.final_table_tsv
    sources = ReviewSources(
        formulation_tsv=formulation_tsv.resolve(),
        seed_rows_tsv=args.seed_rows_tsv.resolve() if args.seed_rows_tsv else None,
        source_index_tsv=args.source_index_tsv.resolve() if args.source_index_tsv else None,
    )
    formulation_rows = read_tsv(sources.formulation_tsv)
    seed_rows = read_tsv(sources.seed_rows_tsv) if sources.seed_rows_tsv else []
    source_index = load_source_index(sources.source_index_tsv)
    cards = build_review_cards(formulation_rows, seed_rows, source_index=source_index, limit=args.limit)
    session_id = f"formulation_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    metadata_path = write_metadata(args.out_dir, session_id, sources, len(cards))

    if args.export_snapshot:
        print(
            json.dumps(
                {
                    "review_session_id": session_id,
                    "card_count": len(cards),
                    "metadata_path": str(metadata_path),
                    "decision_jsonl": str(args.out_dir / DECISIONS_NAME),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    server = ReviewServer(
        (args.host, args.port),
        ReviewHandler,
        cards=cards,
        out_dir=args.out_dir,
        session_id=session_id,
        sources=sources,
        source_index=source_index,
    )
    print(f"Serving formulation review UI at http://{args.host}:{args.port}")
    print(f"Loaded formulation cards: {len(cards)}")
    print(f"Decision ledger: {args.out_dir / DECISIONS_NAME}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping formulation review UI")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
