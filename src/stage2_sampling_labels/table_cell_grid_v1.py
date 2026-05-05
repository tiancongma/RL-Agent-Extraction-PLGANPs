#!/usr/bin/env python3
"""Universal S2-2 table cell-grid structure preservation.

This module intentionally preserves table structure only. It does not decide
whether a column is a factor, measurement, response, formulation identity, or
benchmark-facing field. Downstream semantic/field layers can consume the raw
cell/header grid and make those decisions later.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from src.stage2_sampling_labels.table_row_expansion_v1 import canonical_field_for_header

TABLE_CELL_GRID_TSV_NAME = "table_cell_grid_v1.tsv"
TABLE_CELL_GRID_JSONL_NAME = "table_cell_grid_v1.jsonl"

TABLE_CELL_GRID_COLUMNS = [
    "paper_key",
    "table_id",
    "source_table_asset_id",
    "source_csv_path",
    "normalized_csv_path",
    "source_caption_or_title",
    "row_index",
    "column_index",
    "raw_header_path_json",
    "raw_header_text",
    "raw_cell_value",
    "row_label_candidate",
    "column_label_candidate",
    "cell_kind",
    "structure_status",
    "header_row_index",
    "data_row_index",
    "source_locator",
    "binding_rule",
]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _stringify_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _ensure_matrix(value: Any) -> list[list[str]]:
    rows: list[list[str]] = []
    if not isinstance(value, list):
        return rows
    for row in value:
        if isinstance(row, list):
            rows.append([normalize_text(cell) for cell in row])
    return rows


def _header_structure(payload: dict[str, Any]) -> tuple[list[list[str]], list[str], int]:
    structure = payload.get("header_structure") if isinstance(payload.get("header_structure"), dict) else {}
    header_rows = _ensure_matrix(structure.get("header_rows"))
    flattened = [normalize_text(item) for item in (structure.get("flattened_headers") or [])]
    header_row_count = int(structure.get("header_row_count") or payload.get("header_row_count") or len(header_rows) or 0)
    if not flattened and header_rows:
        column_count = max((len(row) for row in header_rows), default=0)
        for column_index in range(column_count):
            parts = [row[column_index] for row in header_rows if column_index < len(row) and normalize_text(row[column_index])]
            flattened.append(normalize_text(" ".join(parts)))
    return header_rows, flattened, header_row_count


def _normalized_body_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in (payload.get("normalized_rows") or []) if isinstance(row, dict)]
    if rows:
        return rows
    raw_cells = _ensure_matrix(payload.get("raw_cells"))
    _, _, header_row_count = _header_structure(payload)
    out: list[dict[str, Any]] = []
    for zero_based_index, cells in enumerate(raw_cells[header_row_count:], start=header_row_count):
        if not any(normalize_text(cell) for cell in cells):
            continue
        out.append({"row_index": zero_based_index + 1, "cells": cells})
    return out


def _header_path_for_column(header_rows: list[list[str]], column_index: int) -> list[str]:
    return [
        normalize_text(row[column_index])
        for row in header_rows
        if column_index < len(row) and normalize_text(row[column_index])
    ]


def _row_label_candidate(entry: dict[str, Any], cells: list[str]) -> str:
    explicit = normalize_text(entry.get("row_number") or entry.get("row_label") or entry.get("row_id"))
    if explicit:
        return explicit
    for cell in cells:
        if normalize_text(cell):
            return normalize_text(cell)
    return ""


def _match_text(value: Any) -> str:
    text = normalize_text(value).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _table_matches(row: dict[str, str], table_id: str) -> bool:
    wanted = _match_text(table_id)
    if not wanted:
        return True
    return wanted in {_match_text(row.get("table_id")), _match_text(row.get("source_table_asset_id"))}


def _split_row_ordinal_label(row_label: str) -> tuple[str, str]:
    """Return (body_ordinal, semantic_label) for labels like row_02__5.

    Direct table-row expansion uses stable labels such as ``row_02__5`` when the
    first-column formulation label alone is not unique.  The table-cell grid
    preserves both the repeated semantic label (``5``) and the one-based body
    ``data_row_index``.  Parsing the ordinal lets this consumer bind the exact
    row without falling back to raw CSV rebinding or paper-specific matching.
    """
    text = normalize_text(row_label)
    match = re.fullmatch(r"row[_\s-]*0*(\d+)__(.+)", text, flags=re.IGNORECASE)
    if not match:
        return "", text
    ordinal = str(max(int(match.group(1)), 1))
    return ordinal, normalize_text(match.group(2))


def build_grid_cell_bindings_for_row(
    grid_rows: list[dict[str, str]],
    *,
    paper_key: str,
    table_id: str,
    row_label: str = "",
    row_index: str = "",
) -> tuple[list[dict[str, str]], str]:
    """Project table-cell grid structure into row-local metric bindings.

    This consumer is downstream of S2-2 structure recovery: it uses an already
    admitted row identity/locator and only binds cells when exactly one grid row
    candidate matches. It does not create formulation identities.
    """
    paper_key_norm = normalize_text(paper_key)
    row_ordinal_norm, semantic_row_label = _split_row_ordinal_label(row_label)
    label_norm = _match_text(semantic_row_label)
    row_index_norm = normalize_text(row_index)
    candidates = [
        row
        for row in grid_rows
        if normalize_text(row.get("paper_key")) == paper_key_norm
        and _table_matches(row, table_id)
        and (
            (
                label_norm
                and _match_text(row.get("row_label_candidate")) == label_norm
                and (not row_ordinal_norm or normalize_text(row.get("data_row_index")) == row_ordinal_norm)
            )
            or (row_index_norm and normalize_text(row.get("row_index")) == row_index_norm)
        )
    ]
    if not candidates:
        return [], "no_grid_row_candidate"
    candidate_row_keys = {
        (
            normalize_text(row.get("table_id")),
            normalize_text(row.get("source_table_asset_id")),
            normalize_text(row.get("row_index")),
            normalize_text(row.get("row_label_candidate")),
        )
        for row in candidates
    }
    if len(candidate_row_keys) != 1:
        return [], "ambiguous_grid_row_candidates"

    by_field: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        raw_header = normalize_text(row.get("raw_header_text") or row.get("column_label_candidate"))
        raw_value = normalize_text(row.get("raw_cell_value"))
        if not raw_header or not raw_value:
            continue
        canonical_field = canonical_field_for_header(raw_header, paper_key=paper_key_norm)
        if not canonical_field:
            continue
        by_field.setdefault(canonical_field, []).append(row)

    bindings: list[dict[str, str]] = []
    for canonical_field, field_rows in sorted(by_field.items()):
        nonempty_values = {normalize_text(row.get("raw_cell_value")) for row in field_rows if normalize_text(row.get("raw_cell_value"))}
        if len(field_rows) != 1 or len(nonempty_values) != 1:
            continue
        row = field_rows[0]
        bindings.append(
            {
                "source_csv_path": normalize_text(row.get("source_csv_path")),
                "source_table_asset_id": normalize_text(row.get("source_table_asset_id")),
                "source_row_index": normalize_text(row.get("row_index")),
                "source_column_index": normalize_text(row.get("column_index")),
                "raw_header": normalize_text(row.get("raw_header_text") or row.get("column_label_candidate")),
                "canonical_field": canonical_field,
                "raw_cell_value": normalize_text(row.get("raw_cell_value")),
                "source_locator": normalize_text(row.get("source_locator")),
                "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                "ambiguity_status": "unique_grid_header_cell",
            }
        )
    if not bindings:
        return [], "no_canonical_grid_metric_bindings"
    return bindings, "unique_grid_row_binding"


def build_table_cell_grid_from_payload(paper_key: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    """Return one structural row for every body cell in one normalized table payload.

    The output deliberately omits canonical field names and factor/measure roles.
    This is an S2-2 table-structure surface, not a semantic field projection.
    """
    paper_key = normalize_text(paper_key or payload.get("paper_key"))
    table_id = normalize_text(payload.get("table_id") or payload.get("source_table_id"))
    source_table_asset_id = normalize_text(payload.get("source_table_asset_id") or payload.get("table_asset_id"))
    source_csv_path = normalize_text(payload.get("source_csv_path") or payload.get("source_table_reference"))
    normalized_csv_path = normalize_text(payload.get("normalized_csv_path"))
    caption = normalize_text(payload.get("source_caption_or_title") or payload.get("caption") or payload.get("title"))
    header_rows, flattened_headers, header_row_count = _header_structure(payload)
    grid: list[dict[str, str]] = []
    for body_ordinal, entry in enumerate(_normalized_body_rows(payload), start=1):
        cells = [normalize_text(cell) for cell in (entry.get("cells") or [])]
        if not cells:
            continue
        row_index = normalize_text(entry.get("row_index")) or str(header_row_count + body_ordinal)
        row_label = _row_label_candidate(entry, cells)
        for column_index, value in enumerate(cells):
            header_path = _header_path_for_column(header_rows, column_index)
            flattened = flattened_headers[column_index] if column_index < len(flattened_headers) else ""
            raw_header_text = normalize_text(flattened or " ".join(header_path))
            status = "aligned" if raw_header_text else "unlabeled_column"
            locator = f"{table_id or source_table_asset_id}::row_{row_index}::col_{column_index}"
            grid.append({
                "paper_key": paper_key,
                "table_id": table_id,
                "source_table_asset_id": source_table_asset_id,
                "source_csv_path": source_csv_path,
                "normalized_csv_path": normalized_csv_path,
                "source_caption_or_title": caption,
                "row_index": row_index,
                "column_index": str(column_index),
                "raw_header_path_json": _stringify_json(header_path),
                "raw_header_text": raw_header_text,
                "raw_cell_value": value,
                "row_label_candidate": row_label,
                "column_label_candidate": raw_header_text,
                "cell_kind": "body",
                "structure_status": status,
                "header_row_index": "0" if header_rows else "",
                "data_row_index": str(body_ordinal),
                "source_locator": locator,
                "binding_rule": "s2_2_universal_table_cell_grid_v1",
            })
    return grid


def build_table_cell_grid_from_payloads(paper_key: str, payloads: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for payload in payloads:
        if isinstance(payload, dict):
            rows.extend(build_table_cell_grid_from_payload(paper_key, payload))
    return rows


def write_table_cell_grid(output_dir: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / TABLE_CELL_GRID_TSV_NAME
    jsonl_path = output_dir / TABLE_CELL_GRID_JSONL_NAME
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_CELL_GRID_COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in TABLE_CELL_GRID_COLUMNS})
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "table_cell_grid_tsv": str(tsv_path),
        "table_cell_grid_jsonl": str(jsonl_path),
        "table_cell_grid_rows": len(rows),
        "table_cell_grid_tables": len({(row.get("paper_key", ""), row.get("source_table_asset_id", ""), row.get("table_id", "")) for row in rows}),
        "table_cell_grid_papers": len({row.get("paper_key", "") for row in rows if row.get("paper_key", "")}),
    }
