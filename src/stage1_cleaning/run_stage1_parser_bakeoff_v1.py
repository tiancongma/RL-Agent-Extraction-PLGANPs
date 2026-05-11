#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic Stage1 parser bakeoff runner.

This script is intentionally diagnostic-only. It compares current Stage1 parsing
surfaces against optional candidate parsers without writing active Stage1 cleaned
text, key2txt, or runtime parser settings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.stage1_cleaning.pdf2clean import (
    extract_html_native_table_cells,
    extract_marker_markdown_tables,
    extract_marker_pdf_blocks,
    extract_marker_table_caption_line,
    extract_pymupdf_blocks,
    extract_text_from_html,
    load_manifest,
    project_text_from_blocks,
    resolve_stage1_source_path,
    to_repo_rel,
)

SUMMARY_COLUMNS = [
    "paper_key",
    "parser",
    "parser_variant",
    "status",
    "source_type",
    "source_path",
    "text_chars",
    "block_count",
    "table_count",
    "cell_count",
    "warning_count",
    "stage1_table_cell_sidecar_path",
    "stage1_table_cell_sidecar_available",
    "benchmark_valid",
    "notes",
]

WARNING_COLUMNS = ["paper_key", "parser", "status", "warning_code", "warning_detail"]

STAGE1_TABLE_CELL_COLUMNS = [
    "paper_key",
    "source_type",
    "source_path",
    "parser",
    "parser_variant",
    "table_id",
    "table_source_kind",
    "page",
    "bbox_json",
    "caption",
    "caption_binding_rule",
    "caption_source_block_id",
    "continuation_group_id",
    "continuation_binding_rule",
    "noise_class",
    "noise_reason",
    "row_index",
    "col_index",
    "rowspan",
    "colspan",
    "raw_cell_text",
    "normalized_cell_text",
    "is_header_cell",
    "header_scope",
    "header_path_json",
    "row_label_text",
    "column_label_text",
    "source_block_id",
    "source_hash",
    "warnings_json",
]

STAGE1_TABLE_CELL_MANIFEST_COLUMNS = [
    "paper_key",
    "stage1_table_cell_sidecar_path",
    "stage1_table_cell_sidecar_available",
    "cell_count",
    "table_count",
    "source_type",
    "parsers",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha1:{h.hexdigest()}"


def load_scope_keys(scope_keys_path: Path) -> List[str]:
    if not scope_keys_path.exists():
        raise FileNotFoundError(f"scope keys file not found: {scope_keys_path}")
    text = scope_keys_path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    keys: List[str] = []
    first_parts = re.split(r"\t|,", lines[0])
    has_header = first_parts and first_parts[0].strip().lower() in {"key", "paper_key", "doc_key"}
    for line in lines[1 if has_header else 0 :]:
        key = re.split(r"\t|,", line)[0].strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def manifest_rows_by_key(manifest_path: Path) -> Dict[str, Dict[str, str]]:
    df = load_manifest(manifest_path).fillna("")
    rows: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        rec = {str(k): str(v) for k, v in row.to_dict().items()}
        key = (rec.get("key") or rec.get("paper_key") or rec.get("id") or "").strip()
        if key:
            rows[key] = rec
    return rows


def choose_source(row: Dict[str, str], prefer: str = "html") -> tuple[str, Optional[Path], str]:
    pdf_value = next((row.get(c, "").strip() for c in ["pdf", "pdf_path", "pdffile", "file_pdf"] if row.get(c, "").strip()), "")
    html_value = next((row.get(c, "").strip() for c in ["html", "html_path", "htmlfile", "file_html"] if row.get(c, "").strip()), "")
    pdf_path = resolve_stage1_source_path(pdf_value) if pdf_value else None
    html_path = resolve_stage1_source_path(html_value) if html_value else None
    have_pdf = bool(pdf_path and pdf_path.exists() and pdf_path.is_file())
    have_html = bool(html_path and html_path.exists() and html_path.is_file())
    if have_pdf and have_html:
        if prefer.lower() == "pdf":
            return "PDF", pdf_path, pdf_value
        return "HTML", html_path, html_value
    if have_html:
        return "HTML", html_path, html_value
    if have_pdf:
        return "PDF", pdf_path, pdf_value
    return "", None, pdf_value or html_value


def extract_html_cells(html_path: Path, paper_key: str, parser: str) -> List[Dict[str, Any]]:
    cells = extract_html_native_table_cells(html_path, doc_key=paper_key)
    for cell in cells:
        cell["parser"] = parser
        cell["parser_variant"] = "diagnostic_bakeoff_v1"
    return cells


def table_records_from_sidecar(paper_key: str, parser: str, source_type: str, source_path: Path, sidecar: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for table in sidecar.get("tables", []) or []:
        out.append(
            {
                "paper_key": paper_key,
                "parser": parser,
                "parser_variant": "diagnostic_bakeoff_v1",
                "source_type": source_type,
                "source_path": to_repo_rel(source_path) if source_path.exists() else str(source_path),
                "table_id": str(table.get("table_id", "")),
                "format": str(table.get("format", "")),
                "source_file": str(table.get("source_file", "")),
            }
        )
    return out


def block_records(paper_key: str, parser: str, source_type: str, source_path: Path, blocks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for block in blocks:
        out.append(
            {
                "paper_key": paper_key,
                "parser": parser,
                "parser_variant": "diagnostic_bakeoff_v1",
                "source_type": source_type,
                "source_path": to_repo_rel(source_path) if source_path.exists() else str(source_path),
                "block_id": str(block.get("block_id", "")),
                "type": str(block.get("type", "")),
                "page": block.get("page"),
                "order": block.get("order"),
                "table_id": str(block.get("table_id", "")),
                "text": str(block.get("text", "")),
            }
        )
    return out


def _normalized_table_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def bind_marker_table_captions_from_blocks(
    blocks: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    cells: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Bind Marker table records to nearby structural caption blocks.

    Marker sometimes emits captions as separate paragraph blocks immediately
    before table blocks. The raw markdown table parser only sees isolated table
    text in the bakeoff path, so it cannot bind those captions. This helper is
    additive: it fills empty caption metadata only when a table record's
    block_text uniquely matches a structural table block with a preceding
    source-observed Table/Tab. caption line.
    """
    caption_by_table_text: Dict[str, Dict[str, str]] = {}
    pending_caption = ""
    pending_caption_block_id = ""
    for block in sorted(blocks, key=lambda b: int(b.get("order") or 0)):
        block_type = str(block.get("type", ""))
        text = str(block.get("text", ""))
        if block_type == "table":
            key = _normalized_table_text(text)
            if key and pending_caption:
                caption_by_table_text.setdefault(
                    key,
                    {
                        "caption": pending_caption,
                        "caption_binding_rule": "preceding_structural_block_caption",
                        "caption_source_block_id": pending_caption_block_id,
                    },
                )
            pending_caption = ""
            pending_caption_block_id = ""
            continue
        caption = extract_marker_table_caption_line(text)
        if caption:
            pending_caption = caption
            pending_caption_block_id = str(block.get("block_id", ""))
            continue
        if text.strip() and block_type not in {"paragraph", "heading", "caption"}:
            pending_caption = ""
            pending_caption_block_id = ""

    bound_tables: List[Dict[str, Any]] = []
    table_caption_by_id: Dict[str, Dict[str, str]] = {}
    for table in tables:
        rec = dict(table)
        if not str(rec.get("caption", "")).strip():
            binding = caption_by_table_text.get(_normalized_table_text(rec.get("block_text", "")))
            if binding:
                rec.update(binding)
        if str(rec.get("caption", "")).strip():
            table_caption_by_id[str(rec.get("table_id", ""))] = {
                "caption": str(rec.get("caption", "")),
                "caption_binding_rule": str(rec.get("caption_binding_rule", "")),
                "caption_source_block_id": str(rec.get("caption_source_block_id", "")),
            }
        bound_tables.append(rec)

    bound_cells: List[Dict[str, Any]] = []
    for cell in cells:
        rec = dict(cell)
        binding = table_caption_by_id.get(str(rec.get("table_id", "")))
        if binding and not str(rec.get("caption", "")).strip():
            rec.update(binding)
        bound_cells.append(rec)
    return bound_tables, bound_cells


def ok_summary(paper_key: str, parser: str, source_type: str, source_path: Path, text: str, blocks: List[Dict[str, Any]], tables: List[Dict[str, Any]], cells: List[Dict[str, Any]], warnings: List[str]) -> Dict[str, str]:
    return {
        "paper_key": paper_key,
        "parser": parser,
        "parser_variant": "diagnostic_bakeoff_v1",
        "status": "ok",
        "source_type": source_type,
        "source_path": to_repo_rel(source_path) if source_path.exists() else str(source_path),
        "text_chars": str(len(text)),
        "block_count": str(len(blocks)),
        "table_count": str(len(tables)),
        "cell_count": str(len(cells)),
        "warning_count": str(len(warnings)),
        "stage1_table_cell_sidecar_path": "",
        "stage1_table_cell_sidecar_available": "no",
        "benchmark_valid": "no",
        "notes": "diagnostic_only_no_runtime_parser_change",
    }


def diagnostic_summary(paper_key: str, parser: str, status: str, source_type: str, source_path: str, notes: str) -> Dict[str, str]:
    return {
        "paper_key": paper_key,
        "parser": parser,
        "parser_variant": "diagnostic_bakeoff_v1",
        "status": status,
        "source_type": source_type,
        "source_path": source_path,
        "text_chars": "0",
        "block_count": "0",
        "table_count": "0",
        "cell_count": "0",
        "warning_count": "1",
        "stage1_table_cell_sidecar_path": "",
        "stage1_table_cell_sidecar_available": "no",
        "benchmark_valid": "no",
        "notes": notes,
    }


def run_current_candidate(paper_key: str, source_type: str, source_path: Path, table_dir_value: str) -> tuple[Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    if source_type == "HTML":
        result = extract_text_from_html(source_path, table_dir_value=table_dir_value)
        sidecar = result["sidecar"]
        blocks = block_records(paper_key, "current", source_type, source_path, sidecar.get("blocks", []) or [])
        tables = table_records_from_sidecar(paper_key, "current", source_type, source_path, sidecar)
        cells = extract_html_cells(source_path, paper_key, "current")
        warnings = [str(w) for w in sidecar.get("metadata", {}).get("warnings", []) or []]
        summary = ok_summary(paper_key, "current", source_type, source_path, str(result.get("text", "")), blocks, tables, cells, warnings)
        return summary, blocks, tables, cells, warnings
    if source_type == "PDF":
        raw_blocks, page_count = extract_pymupdf_blocks(source_path, max_pages=0)
        finalized = []
        # Reuse block shape without writing Stage1 outputs.
        for idx, block in enumerate(raw_blocks, start=1):
            rec = dict(block)
            rec.setdefault("block_id", f"b{idx:04d}")
            rec.setdefault("order", idx)
            finalized.append(rec)
        text = project_text_from_blocks(finalized)
        blocks = block_records(paper_key, "current", source_type, source_path, finalized)
        tables: List[Dict[str, Any]] = []
        cells: List[Dict[str, Any]] = []
        warnings = [f"pymupdf_page_count:{page_count}"]
        summary = ok_summary(paper_key, "current", source_type, source_path, text, blocks, tables, cells, warnings)
        return summary, blocks, tables, cells, warnings
    raise ValueError(f"unsupported source_type for current parser: {source_type}")


def run_marker_candidate(paper_key: str, source_type: str, source_path: Path) -> tuple[Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    if source_type != "PDF":
        note = "marker_candidate_supports_pdf_only"
        return diagnostic_summary(paper_key, "marker", "marker_unsupported_source", source_type, str(source_path), note), [], [], [], [note]
    try:
        raw_blocks, page_count, parser_warnings, marker_tables = extract_marker_pdf_blocks(source_path)
    except Exception as exc:
        status = "marker_unavailable" if "marker_not_available" in str(exc) else "marker_failed"
        note = str(exc)
        return diagnostic_summary(paper_key, "marker", status, source_type, str(source_path), note), [], [], [], [f"{status}:{note}"]
    finalized = []
    for idx, block in enumerate(raw_blocks, start=1):
        rec = dict(block)
        rec.setdefault("block_id", f"b{idx:04d}")
        rec.setdefault("order", idx)
        finalized.append(rec)
    text = project_text_from_blocks(finalized)
    blocks = block_records(paper_key, "marker", source_type, source_path, finalized)
    marker_table_records: List[Dict[str, Any]] = []
    marker_cells: List[Dict[str, Any]] = []
    for idx, table_text in enumerate(marker_tables, start=1):
        parsed_tables, parsed_cells = extract_marker_markdown_tables(
            rendered_text=table_text,
            paper_key=paper_key,
            source_path=source_path,
            parser="marker",
            parser_variant="diagnostic_bakeoff_v1",
        )
        if parsed_tables:
            table_id = f"t{len(marker_table_records) + 1:03d}"
            old_id = str(parsed_tables[0].get("table_id", ""))
            parsed_tables[0]["table_id"] = table_id
            parsed_tables[0]["source_path"] = to_repo_rel(source_path) if source_path.exists() else str(source_path)
            parsed_tables[0]["source_file"] = to_repo_rel(source_path) if source_path.exists() else str(source_path)
            marker_table_records.append(parsed_tables[0])
            for cell in parsed_cells:
                if str(cell.get("table_id", "")) == old_id:
                    cell["table_id"] = table_id
                    cell["source_block_id"] = table_id
                marker_cells.append(cell)
        else:
            marker_table_records.append(
                {
                    "paper_key": paper_key,
                    "parser": "marker",
                    "parser_variant": "diagnostic_bakeoff_v1",
                    "source_type": source_type,
                    "source_path": to_repo_rel(source_path) if source_path.exists() else str(source_path),
                    "table_id": f"t{len(marker_table_records) + 1:03d}",
                    "format": "marker_text",
                    "source_file": to_repo_rel(source_path) if source_path.exists() else str(source_path),
                    "block_text": table_text,
                }
            )
    marker_table_records, marker_cells = bind_marker_table_captions_from_blocks(finalized, marker_table_records, marker_cells)
    tables = marker_table_records
    cells = marker_cells
    warnings = [str(w) for w in parser_warnings] + [f"marker_page_count:{page_count}"]
    summary = ok_summary(paper_key, "marker", source_type, source_path, text, blocks, tables, cells, warnings)
    return summary, blocks, tables, cells, warnings


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _json_array_text(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    text = str(value or "").strip()
    if not text:
        return "[]"
    try:
        parsed = json.loads(text)
    except Exception:
        return json.dumps([text], ensure_ascii=False)
    if isinstance(parsed, list):
        return json.dumps(parsed, ensure_ascii=False)
    return json.dumps([parsed], ensure_ascii=False)


def _json_object_text(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value or "").strip()
    if not text:
        return "{}"
    try:
        parsed = json.loads(text)
    except Exception:
        return json.dumps({"raw": text}, ensure_ascii=False, sort_keys=True)
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    return json.dumps({"raw": parsed}, ensure_ascii=False, sort_keys=True)


def normalize_stage1_table_cell_row(cell: Dict[str, Any]) -> Dict[str, str]:
    """Return one source-faithful Stage1 table-cell sidecar row."""

    row: Dict[str, str] = {}
    for column in STAGE1_TABLE_CELL_COLUMNS:
        value = cell.get(column, "")
        if column in {"header_path_json", "warnings_json"}:
            row[column] = _json_array_text(value)
        elif column == "bbox_json":
            row[column] = _json_object_text(value)
        else:
            row[column] = str(value if value is not None else "")
    return row


def _sidecar_sort_key(row: Dict[str, str]) -> tuple[str, str, str, int, int]:
    def as_int(value: str) -> int:
        try:
            return int(str(value))
        except Exception:
            return 0

    return (
        row.get("paper_key", ""),
        row.get("parser", ""),
        row.get("table_id", ""),
        as_int(row.get("row_index", "")),
        as_int(row.get("col_index", "")),
    )


def write_stage1_cell_sidecars(out_dir: Path, cell_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Write diagnostic run-scoped per-paper Stage1 table-cell sidecars."""

    sidecar_root = out_dir.expanduser().resolve() / "tables_cell_sidecar"
    normalized_rows = [normalize_stage1_table_cell_row(row) for row in cell_rows]
    normalized_rows.sort(key=_sidecar_sort_key)
    by_paper: Dict[str, List[Dict[str, str]]] = {}
    for row in normalized_rows:
        paper_key = row.get("paper_key", "").strip()
        if not paper_key:
            continue
        by_paper.setdefault(paper_key, []).append(row)

    manifest_rows: List[Dict[str, str]] = []
    if not by_paper:
        return manifest_rows
    sidecar_root.mkdir(parents=True, exist_ok=True)
    for paper_key in sorted(by_paper):
        paper_rows = by_paper[paper_key]
        paper_dir = sidecar_root / paper_key
        paper_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = paper_dir / "stage1_table_cells_v1.jsonl"
        with sidecar_path.open("w", encoding="utf-8") as f:
            for row in paper_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest_rows.append(
            {
                "paper_key": paper_key,
                "stage1_table_cell_sidecar_path": str(sidecar_path),
                "stage1_table_cell_sidecar_available": "yes",
                "cell_count": str(len(paper_rows)),
                "table_count": str(len({row.get("table_id", "") for row in paper_rows if row.get("table_id", "")})),
                "source_type": ";".join(sorted({row.get("source_type", "") for row in paper_rows if row.get("source_type", "")})),
                "parsers": ";".join(sorted({row.get("parser", "") for row in paper_rows if row.get("parser", "")})),
            }
        )
    write_tsv(sidecar_root / "stage1_table_cells_manifest_v1.tsv", manifest_rows, STAGE1_TABLE_CELL_MANIFEST_COLUMNS)
    return manifest_rows


def write_tsv(path: Path, rows: List[Dict[str, str]], columns: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_run_context(out_dir: Path, manifest_path: Path, scope_keys_path: Path, parser_selection: str, summary_rows: List[Dict[str, str]]) -> None:
    ok = sum(1 for row in summary_rows if row.get("status") == "ok")
    content = "\n".join(
        [
            "# Stage1 Parser Bakeoff Diagnostic RUN_CONTEXT",
            "",
            f"generated_at: {utc_now_iso()}",
            "diagnostic_only: yes",
            "benchmark_valid: no",
            "live_llm_calls: no",
            "active_run_json_modified: no",
            "runtime_parser_behavior_modified: no",
            f"script: src/stage1_cleaning/run_stage1_parser_bakeoff_v1.py",
            f"manifest_path: {manifest_path}",
            f"manifest_sha1: {sha1_file(manifest_path) if manifest_path.exists() else ''}",
            f"scope_keys_path: {scope_keys_path}",
            f"scope_keys_sha1: {sha1_file(scope_keys_path) if scope_keys_path.exists() else ''}",
            f"parser_selection: {parser_selection}",
            f"output_dir: {out_dir}",
            f"summary_rows: {len(summary_rows)}",
            f"ok_rows: {ok}",
            "",
            "Outputs:",
            "- parser_bakeoff_summary_v1.tsv",
            "- parser_bakeoff_blocks_v1.jsonl",
            "- parser_bakeoff_tables_v1.jsonl",
            "- parser_bakeoff_cells_v1.jsonl",
            "- tables_cell_sidecar/<paper_key>/stage1_table_cells_v1.jsonl",
            "- tables_cell_sidecar/stage1_table_cells_manifest_v1.tsv",
            "- parser_bakeoff_warnings_v1.tsv",
            "",
        ]
    )
    (out_dir / "RUN_CONTEXT.md").write_text(content, encoding="utf-8")


def run_bakeoff(manifest_path: Path, scope_keys_path: Path, out_dir: Path, parser_selection: str = "all", prefer: str = "html") -> Dict[str, Any]:
    if parser_selection not in {"current", "marker", "all"}:
        raise ValueError("parser_selection must be current, marker, or all")
    manifest_path = manifest_path.expanduser().resolve()
    scope_keys_path = scope_keys_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = load_scope_keys(scope_keys_path)
    rows_by_key = manifest_rows_by_key(manifest_path)
    parsers = ["current", "marker"] if parser_selection == "all" else [parser_selection]

    summary_rows: List[Dict[str, str]] = []
    block_rows: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, Any]] = []
    cell_rows: List[Dict[str, Any]] = []
    warning_rows: List[Dict[str, str]] = []

    for key in keys:
        row = rows_by_key.get(key)
        if row is None:
            for parser in parsers:
                note = "scope_key_missing_from_manifest"
                summary_rows.append(diagnostic_summary(key, parser, "missing_manifest_row", "", "", note))
                warning_rows.append({"paper_key": key, "parser": parser, "status": "missing_manifest_row", "warning_code": "missing_manifest_row", "warning_detail": note})
            continue
        source_type, source_path, raw_source_value = choose_source(row, prefer=prefer)
        for parser in parsers:
            if not source_path or not source_type:
                note = "no_resolved_input_source"
                summary_rows.append(diagnostic_summary(key, parser, "missing_source", source_type, raw_source_value, note))
                warning_rows.append({"paper_key": key, "parser": parser, "status": "missing_source", "warning_code": "missing_source", "warning_detail": note})
                continue
            try:
                if parser == "current":
                    summary, blocks, tables, cells, warnings = run_current_candidate(key, source_type, source_path, row.get("table_dir", ""))
                else:
                    summary, blocks, tables, cells, warnings = run_marker_candidate(key, source_type, source_path)
            except Exception as exc:
                status = f"{parser}_failed"
                note = f"{type(exc).__name__}: {exc}"
                summary, blocks, tables, cells, warnings = diagnostic_summary(key, parser, status, source_type, str(source_path), note), [], [], [], [note]
            summary_rows.append(summary)
            block_rows.extend(blocks)
            table_rows.extend(tables)
            cell_rows.extend(cells)
            for warning in warnings:
                warning_rows.append(
                    {
                        "paper_key": key,
                        "parser": parser,
                        "status": summary["status"],
                        "warning_code": str(warning).split(":", 1)[0],
                        "warning_detail": str(warning),
                    }
                )

    sidecar_manifest_rows = write_stage1_cell_sidecars(out_dir, cell_rows)
    sidecar_by_paper = {row.get("paper_key", ""): row for row in sidecar_manifest_rows}
    for row in summary_rows:
        paper_sidecar = sidecar_by_paper.get(row.get("paper_key", ""))
        if paper_sidecar:
            row["stage1_table_cell_sidecar_path"] = paper_sidecar.get("stage1_table_cell_sidecar_path", "")
            row["stage1_table_cell_sidecar_available"] = paper_sidecar.get("stage1_table_cell_sidecar_available", "yes") or "yes"

    write_tsv(out_dir / "parser_bakeoff_summary_v1.tsv", summary_rows, SUMMARY_COLUMNS)
    write_jsonl(out_dir / "parser_bakeoff_blocks_v1.jsonl", block_rows)
    write_jsonl(out_dir / "parser_bakeoff_tables_v1.jsonl", table_rows)
    write_jsonl(out_dir / "parser_bakeoff_cells_v1.jsonl", cell_rows)
    write_tsv(out_dir / "parser_bakeoff_warnings_v1.tsv", warning_rows, WARNING_COLUMNS)
    write_run_context(out_dir, manifest_path, scope_keys_path, parser_selection, summary_rows)
    return {"status": "completed", "out_dir": str(out_dir), "summary_rows": len(summary_rows), "ok_rows": sum(1 for row in summary_rows if row.get("status") == "ok")}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run diagnostic-only Stage1 parser bakeoff without changing runtime parser outputs.")
    ap.add_argument("--manifest", type=Path, required=True, help="Explicit manifest TSV/CSV path.")
    ap.add_argument("--scope-keys", type=Path, required=True, help="Explicit key list/TSV path.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Explicit diagnostic output directory.")
    ap.add_argument("--parser", choices=["current", "marker", "all"], default="all", help="Parser candidate(s) to run.")
    ap.add_argument("--prefer", choices=["html", "pdf"], default="html", help="Source preference when both source files exist.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_bakeoff(
        manifest_path=args.manifest,
        scope_keys_path=args.scope_keys,
        out_dir=args.out_dir,
        parser_selection=args.parser,
        prefer=args.prefer,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
