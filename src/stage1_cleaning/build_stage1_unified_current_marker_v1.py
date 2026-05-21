#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage1_unified_current_marker_v1.py

Purpose
- Build one governed Stage1 document surface that projects current HTML/PDF
  clean text, current structure, frozen Marker PDF structure/text when available,
  and current Stage1 table-cell sidecars into a single downstream-consumable
  interface.

Contract
- Inputs are existing governed Stage1 indexes/surfaces:
    data/cleaned/index/manifest_current.tsv
    data/cleaned/index/key2txt.tsv
    data/cleaned/index/key2structure.tsv
    data/cleaned/index/key2marker_pdf_v1.tsv
    data/cleaned/index/stage1_table_cells_manifest_v1.tsv
- Outputs are additive governed Stage1 artifacts:
    data/cleaned/content/stage1_unified_current_marker_v1/<paper_key>/unified_clean_text_v1.md
    data/cleaned/content/stage1_unified_current_marker_v1/<paper_key>/unified_structure_v1.json
    data/cleaned/index/key2stage1_unified_current_marker_v1.tsv
    data/cleaned/index/stage1_unified_current_marker_manifest_v1.tsv
    data/cleaned/index/stage1_unified_current_marker_summary_v1.tsv

Principles
- No Marker rerun. Frozen Marker output is consumed only if already present.
- No PLGA semantic inference, no Stage2/Stage5 execution, no benchmark claim.
- HTML current clean text/structure is preferred when both HTML and PDF current
  Stage1 bindings exist for the same paper. PDF current text remains the
  compatibility fallback; Marker text is additive for PDF records only when
  frozen Marker output exists.
- The emitted JSON exposes a Stage2-friendly top-level `blocks` list and a
  manifest-friendly `text_path`, `structure_path`, and
  `stage1_table_cell_sidecar_path` binding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "src" / "utils" / "paths.py").exists():
            sys.path.insert(0, str(p))
            return


_bootstrap_import_paths()
from src.utils import paths  # noqa: E402

SCRIPT_CONTRACT_VERSION = "stage1_unified_current_marker_v1"

SUMMARY_FIELDS = [
    "paper_key",
    "status",
    "source_type",
    "current_text_path",
    "current_structure_path",
    "marker_text_path",
    "marker_structure_path",
    "unified_text_path",
    "unified_structure_path",
    "stage1_table_cell_sidecar_path",
    "stage1_table_cell_sidecar_available",
    "current_text_sha256",
    "current_structure_sha256",
    "marker_text_sha256",
    "marker_structure_sha256",
    "unified_text_sha256",
    "unified_structure_sha256",
    "current_text_length",
    "marker_text_length",
    "unified_text_length",
    "current_block_count",
    "marker_block_count",
    "unified_block_count",
    "table_dir",
    "table_available",
    "table_cell_count",
    "table_count",
    "table_authority_inherited",
    "table_authority_inheritance_source",
    "contract_version",
    "note",
]

KEY2_FIELDS = [
    "paper_key",
    "text_path",
    "structure_path",
    "text_source_type",
    "text_available",
    "structure_available",
    "table_dir",
    "table_available",
    "stage1_table_cell_sidecar_path",
    "stage1_table_cell_sidecar_available",
    "table_cell_count",
    "table_count",
    "table_authority_inherited",
    "table_authority_inheritance_source",
    "unified_text_sha256",
    "unified_structure_sha256",
    "status",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(paths.PROJECT_ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def resolve_repo_path(value: str | Path) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        return p
    return paths.PROJECT_ROOT / p


def read_scope(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    keys: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.readline()
        f.seek(0)
        if "\t" in sample and any(h in sample.split("\t") for h in ["paper_key", "key", "zotero_key"]):
            for row in csv.DictReader(f, delimiter="\t"):
                key = (row.get("paper_key") or row.get("key") or row.get("zotero_key") or "").strip()
                if key:
                    keys.add(key)
            return keys
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#") and s not in {"paper_key", "key", "zotero_key"}:
                keys.add(s.split("\t")[0].split(",")[0].strip())
    return keys


def load_manifest(path: Path, scope: set[str] | None) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row.get("paper_key") or row.get("key") or row.get("zotero_key") or "").strip()
            if key and (scope is None or key in scope):
                out[key] = row
    return out


def current_binding_rank(value: str) -> int:
    """Rank duplicate current Stage1 bindings by source preference.

    HTML is preferred over PDF when both bindings exist for the same key.
    This preserves the full-corpus Stage1 handoff design: HTML-native text,
    structure, and table sidecars are used where available; PDF remains the
    fallback and may receive additive frozen Marker content later in this script.
    """
    v = (value or "").lower()
    if ".html" in v or v.endswith("html.txt") or v.endswith("html.json"):
        return 2
    if ".pdf" in v or v.endswith("pdf.txt") or v.endswith("pdf.json"):
        return 1
    return 0


def prefer_current_binding(existing: str, candidate: str) -> str:
    if not existing:
        return candidate
    if current_binding_rank(candidate) > current_binding_rank(existing):
        return candidate
    return existing


def prefer_current_structure_row(existing: dict[str, str] | None, candidate: dict[str, str]) -> dict[str, str]:
    if not existing:
        return candidate
    existing_value = existing.get("structure_path") or existing.get("txt_path") or existing.get("text_path") or ""
    candidate_value = candidate.get("structure_path") or candidate.get("txt_path") or candidate.get("text_path") or ""
    if current_binding_rank(candidate_value) > current_binding_rank(existing_value):
        return candidate
    return existing


def load_key2txt(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.readline()
        f.seek(0)
        if sample.startswith("key\t") or sample.startswith("paper_key\t"):
            for row in csv.DictReader(f, delimiter="\t"):
                key = (row.get("paper_key") or row.get("key") or "").strip()
                val = (row.get("txt_path") or row.get("text_path") or "").strip()
                if key and val:
                    out[key] = prefer_current_binding(out.get(key, ""), val)
        else:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0] and parts[1]:
                    out[parts[0]] = prefer_current_binding(out.get(parts[0], ""), parts[1])
    return out


def load_key2structure(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row.get("key") or row.get("paper_key") or "").strip()
            if key:
                out[key] = prefer_current_structure_row(out.get(key), row)
    return out


def load_tsv_by_key(path: Path, key_field: str = "paper_key") -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row.get(key_field) or row.get("paper_key") or row.get("key") or "").strip()
            if key:
                out[key] = row
    return out


def load_table_authority_rows(paths_: list[Path]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for path in paths_:
        if not path.exists():
            raise FileNotFoundError(f"Table authority manifest not found: {path}")
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                key = (row.get("paper_key") or row.get("key") or row.get("zotero_key") or "").strip()
                if not key:
                    continue
                table_dir = str(row.get("table_dir") or "").strip()
                sidecar_path = str(row.get("stage1_table_cell_sidecar_path") or "").strip()
                if table_dir or sidecar_path:
                    inherited = dict(row)
                    inherited["_table_authority_manifest_source"] = repo_rel(path)
                    out[key] = inherited
    return out


def merge_table_authority_row(
    manifest_row: dict[str, str],
    authority_row: dict[str, str] | None,
) -> dict[str, str]:
    merged = dict(manifest_row)
    merged.setdefault("table_authority_inherited", "no")
    merged.setdefault("table_authority_inheritance_source", "")
    if not authority_row:
        return merged

    current_table_dir = str(merged.get("table_dir") or "").strip()
    current_sidecar = str(merged.get("stage1_table_cell_sidecar_path") or "").strip()
    authority_table_dir = str(authority_row.get("table_dir") or "").strip()
    authority_sidecar = str(authority_row.get("stage1_table_cell_sidecar_path") or "").strip()

    inherited_any = False
    if not current_table_dir and authority_table_dir:
        merged["table_dir"] = authority_table_dir
        merged["table_available"] = authority_row.get("table_available") or "yes"
        inherited_any = True
    if not current_sidecar and authority_sidecar:
        merged["stage1_table_cell_sidecar_path"] = authority_sidecar
        merged["stage1_table_cell_sidecar_available"] = (
            authority_row.get("stage1_table_cell_sidecar_available") or "yes"
        )
        inherited_any = True

    for field in ["table_cell_count", "table_count"]:
        if not str(merged.get(field) or "").strip() and str(authority_row.get(field) or "").strip():
            merged[field] = str(authority_row.get(field) or "").strip()
            inherited_any = True

    if inherited_any:
        merged["table_authority_inherited"] = "yes"
        merged["table_authority_inheritance_source"] = authority_row.get("_table_authority_manifest_source", "")
    return merged


def load_json_if_exists(rel_or_abs: str) -> tuple[dict[str, Any], str, str]:
    if not rel_or_abs:
        return {}, "", ""
    p = resolve_repo_path(rel_or_abs)
    if not p.exists():
        return {}, "", ""
    return json.loads(p.read_text(encoding="utf-8", errors="replace")), repo_rel(p), sha256_file(p)


def infer_source_type(row: dict[str, str], current_structure: dict[str, Any]) -> str:
    # Prefer the actual selected current binding over manifest-level defaults.
    # `manifest_current.tsv` may still say `text_source_type=pdf` for rows where
    # key2txt/key2structure have both PDF and HTML derivatives; the unified
    # Stage1 handoff is HTML-first when the selected binding is HTML.
    txt = row.get("text_path") or row.get("txt_path") or ""
    if ".html" in txt.lower():
        return "HTML"
    if ".pdf" in txt.lower():
        return "PDF"
    struct_source_type = str(current_structure.get("source_type", "")).strip()
    if struct_source_type:
        return struct_source_type.upper()
    manifest_source_type = str(row.get("text_source_type", "")).strip()
    if manifest_source_type:
        return manifest_source_type.upper()
    if row.get("html"):
        return "HTML"
    if row.get("pdf"):
        return "PDF"
    return "UNKNOWN"


def normalize_current_blocks(blocks: list[Any], source_type: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, block in enumerate(blocks or [], start=1):
        if not isinstance(block, dict):
            continue
        text = str(block.get("text") or block.get("block_text") or block.get("raw_text") or "")
        out.append(
            {
                "block_id": str(block.get("block_id") or f"current_b{i:04d}"),
                "source_parser": "current",
                "source_type": source_type,
                "type": str(block.get("type") or block.get("block_type") or "paragraph").lower(),
                "page": block.get("page"),
                "order": block.get("order") if block.get("order") is not None else i,
                "text": text,
                "block_text": text,
                "table_id": str(block.get("table_id") or ""),
                "bbox": block.get("bbox") or block.get("bbox_json"),
                "section_hierarchy": block.get("section_hierarchy") or {},
                "source_block_id": str(block.get("block_id") or ""),
            }
        )
    return out


def normalize_marker_blocks(blocks: list[Any], start_order: int = 1) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, block in enumerate(blocks or [], start=1):
        if not isinstance(block, dict):
            continue
        text = str(block.get("text") or block.get("block_text") or block.get("html") or "")
        block_type = str(block.get("type") or block.get("block_type") or block.get("kind") or "").strip()
        if not block_type:
            block_type = "paragraph"
        out.append(
            {
                "block_id": str(block.get("block_id") or f"marker_b{i:04d}"),
                "source_parser": "marker_pdf",
                "source_type": "PDF",
                "type": block_type.lower(),
                "page": block.get("page") if block.get("page") is not None else block.get("page_index"),
                "order": start_order + i - 1,
                "text": text,
                "block_text": text,
                "table_id": str(block.get("table_id") or ""),
                "bbox": block.get("bbox") or block.get("polygon"),
                "section_hierarchy": block.get("section_hierarchy") or {},
                "source_block_id": str(block.get("block_id") or ""),
            }
        )
    return out


def bbox_values(block: dict[str, Any]) -> list[float]:
    raw = block.get("bbox") or []
    if isinstance(raw, list) and len(raw) >= 4:
        try:
            return [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
        except Exception:
            return []
    return []


def bbox_contains(outer: list[float], inner: list[float], tolerance: float = 3.0) -> bool:
    if len(outer) < 4 or len(inner) < 4:
        return False
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def cluster_coordinate(value: float, clusters: list[float], tolerance: float) -> int:
    for index, existing in enumerate(clusters):
        if abs(existing - value) <= tolerance:
            clusters[index] = (existing + value) / 2.0
            return index + 1
    clusters.append(value)
    clusters.sort()
    return clusters.index(value) + 1


def find_nearby_marker_caption(blocks: list[dict[str, Any]], table_index: int) -> str:
    for offset in range(1, 4):
        before_index = table_index - offset
        if before_index >= 0:
            before = blocks[before_index]
            if str(before.get("block_type") or "").lower() == "caption":
                text = str(before.get("text") or "").strip()
                if text:
                    return text
        after_index = table_index + offset
        if after_index < len(blocks):
            after = blocks[after_index]
            if str(after.get("block_type") or "").lower() == "caption":
                text = str(after.get("text") or "").strip()
                if text:
                    return text
    return ""


def marker_cells_to_matrix(table_cells: list[dict[str, Any]]) -> tuple[list[list[str]], list[dict[str, Any]]]:
    if not table_cells:
        return [], []
    row_clusters: list[float] = []
    col_clusters: list[float] = []
    positioned: list[tuple[int, int, dict[str, Any]]] = []
    for cell in sorted(table_cells, key=lambda item: (bbox_values(item)[1] if bbox_values(item) else 0.0, bbox_values(item)[0] if bbox_values(item) else 0.0)):
        bbox = bbox_values(cell)
        if len(bbox) < 4:
            continue
        height = max(1.0, bbox[3] - bbox[1])
        width = max(1.0, bbox[2] - bbox[0])
        row_index = cluster_coordinate(bbox[1], row_clusters, max(2.0, height * 0.45))
        col_index = cluster_coordinate(bbox[0], col_clusters, max(2.0, width * 0.25))
        positioned.append((row_index, col_index, cell))
    if not positioned:
        return [], []
    row_count = max(row for row, _col, _cell in positioned)
    col_count = max(col for _row, col, _cell in positioned)
    matrix = [["" for _ in range(col_count)] for _ in range(row_count)]
    normalized_cells: list[dict[str, Any]] = []
    for row_index, col_index, cell in positioned:
        text = str(cell.get("text") or "").strip()
        current = matrix[row_index - 1][col_index - 1]
        matrix[row_index - 1][col_index - 1] = f"{current} {text}".strip() if current and text else (current or text)
        normalized_cells.append(
            {
                "row_index": row_index,
                "col_index": col_index,
                "raw_cell_text": text,
                "normalized_cell_text": text,
                "bbox_json": {"bbox": bbox_values(cell)},
                "source_block_id": str(cell.get("block_id") or ""),
            }
        )
    return matrix, normalized_cells


def extract_marker_table_assets_from_blocks(
    *,
    paper_key: str,
    marker_blocks: list[dict[str, Any]],
    paper_out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str, str, str]:
    """Promote Marker PDF table blocks into Stage1 execution-grade assets.

    This is structural promotion only: it preserves Marker table geometry as
    CSV/JSONL authority without deciding whether a table is formulation-bearing.
    """
    table_blocks = [
        (index, block)
        for index, block in enumerate(marker_blocks)
        if str(block.get("block_type") or "").lower() == "table"
    ]
    cell_blocks = [
        block
        for block in marker_blocks
        if str(block.get("block_type") or "").lower() == "tablecell"
    ]
    if not table_blocks or not cell_blocks:
        return [], [], "", "no", "0", "0"

    tables_dir = paper_out / "tables"
    sidecar_path = paper_out / "stage1_table_cells_v1.jsonl"
    tables: list[dict[str, Any]] = []
    sidecar_rows: list[dict[str, Any]] = []
    used_cell_ids: set[str] = set()

    for ordinal, (block_index, table_block) in enumerate(table_blocks, start=1):
        table_bbox = bbox_values(table_block)
        page_index = table_block.get("page_index")
        matched_cells = [
            cell
            for cell in cell_blocks
            if str(cell.get("block_id") or "") not in used_cell_ids
            and cell.get("page_index") == page_index
            and bbox_contains(table_bbox, bbox_values(cell))
        ]
        if not matched_cells:
            continue
        matrix, cell_rows = marker_cells_to_matrix(matched_cells)
        if not matrix:
            continue
        tables_dir.mkdir(parents=True, exist_ok=True)
        table_id = f"Table {ordinal}"
        csv_name = f"{paper_key}__marker_table_{ordinal:02d}.csv"
        csv_path = tables_dir / csv_name
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(matrix)
        caption = find_nearby_marker_caption(marker_blocks, block_index)
        table_record = {
            "table_id": table_id,
            "csv_path": csv_name,
            "caption_or_title": caption,
            "caption_binding_source": "marker_structural_block_adjacency" if caption else "",
            "caption_binding_status": "trusted_manifest" if caption else "unbound",
            "caption_binding_rule": "nearby_marker_caption_block" if caption else "",
            "table_source_kind": "marker_pdf_table_block",
            "source_table_reference": str(table_block.get("block_id") or ""),
            "source_table_asset_local_path": repo_rel(csv_path),
            "parser": "marker_pdf",
        }
        tables.append(table_record)
        for row in cell_rows:
            source_block_id = row.pop("source_block_id", "")
            sidecar_rows.append(
                {
                    "paper_key": paper_key,
                    "parser": "marker_pdf",
                    "source_type": "PDF",
                    "table_id": table_id,
                    "row_index": str(row["row_index"]),
                    "col_index": str(row["col_index"]),
                    "rowspan": "1",
                    "colspan": "1",
                    "raw_cell_text": row["raw_cell_text"],
                    "normalized_cell_text": row["normalized_cell_text"],
                    "bbox_json": row["bbox_json"],
                    "header_path_json": [],
                    "caption": caption,
                    "caption_binding_rule": table_record["caption_binding_rule"],
                    "continuation_group_id": "",
                    "noise_class": "",
                    "noise_reason": "",
                    "source_hash": "",
                    "source_block_id": source_block_id,
                    "warnings_json": [],
                }
            )
            if source_block_id:
                used_cell_ids.add(source_block_id)

    if not tables:
        return [], [], "", "no", "0", "0"
    manifest_payload = {
        "paper_key": paper_key,
        "contract_version": "stage1_marker_pdf_table_block_promotion_v1",
        "tables": tables,
        "selected_table_files": [item["csv_path"] for item in tables],
    }
    (tables_dir / "tables_manifest.json").write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with sidecar_path.open("w", encoding="utf-8") as handle:
        for row in sidecar_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return tables, sidecar_rows, repo_rel(tables_dir), "yes", str(len(sidecar_rows)), str(len(tables))


def extract_table_assets_from_stage1_sidecar(
    *,
    paper_key: str,
    sidecar_path: str,
    paper_out: Path,
) -> tuple[list[dict[str, Any]], str, str]:
    path = resolve_repo_path(sidecar_path)
    if not path.exists():
        return [], "", "no"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            if str(item.get("paper_key") or "").strip() not in {"", paper_key}:
                continue
            rows.append(item)
    if not rows:
        return [], "", "no"
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        table_id = str(row.get("table_id") or "").strip()
        if table_id:
            grouped.setdefault(table_id, []).append(row)
    if not grouped:
        return [], "", "no"
    tables_dir = paper_out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_records: list[dict[str, Any]] = []
    for ordinal, table_id in enumerate(sorted(grouped), start=1):
        table_rows = grouped[table_id]
        coords: dict[tuple[int, int], str] = {}
        max_row = 0
        max_col = 0
        for row in table_rows:
            row_index_text = str(row.get("row_index") or "").strip()
            col_index_text = str(row.get("col_index") or "").strip()
            if not row_index_text.isdigit() or not col_index_text.isdigit():
                continue
            row_index = int(row_index_text)
            col_index = int(col_index_text)
            value = str(row.get("normalized_cell_text") or row.get("raw_cell_text") or "").strip()
            coords[(row_index, col_index)] = value
            max_row = max(max_row, row_index)
            max_col = max(max_col, col_index)
        if max_row <= 0 or max_col <= 0:
            continue
        matrix = [
            [coords.get((row_index, col_index), "") for col_index in range(1, max_col + 1)]
            for row_index in range(1, max_row + 1)
        ]
        csv_name = f"{paper_key}__sidecar_table_{ordinal:02d}.csv"
        csv_path = tables_dir / csv_name
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(matrix)
        caption = next((str(row.get("caption") or "").strip() for row in table_rows if str(row.get("caption") or "").strip()), "")
        table_records.append(
            {
                "table_id": table_id,
                "csv_path": csv_name,
                "caption_or_title": caption,
                "caption_binding_source": "stage1_table_cell_sidecar" if caption else "",
                "caption_binding_status": "trusted_manifest" if caption else "unbound",
                "caption_binding_rule": "sidecar_caption" if caption else "",
                "table_source_kind": "stage1_table_cell_sidecar",
                "source_table_reference": sidecar_path,
                "source_table_asset_local_path": repo_rel(csv_path),
                "parser": "stage1_table_cell_sidecar",
            }
        )
    if not table_records:
        return [], "", "no"
    manifest_payload = {
        "paper_key": paper_key,
        "contract_version": "stage1_sidecar_table_dir_projection_v1",
        "tables": table_records,
        "selected_table_files": [item["csv_path"] for item in table_records],
    }
    (tables_dir / "tables_manifest.json").write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return table_records, repo_rel(tables_dir), "yes"


def html_pdf_sibling_supplement_path(current_text_path: Path) -> Path | None:
    name = current_text_path.name
    if not name.endswith(".html.txt"):
        return None
    candidate = current_text_path.with_name(name[: -len(".html.txt")] + ".pdf.txt")
    return candidate if candidate.exists() else None


def canonical_pdf_clean_text_supplement_path(key: str) -> Path | None:
    candidate = paths.PROJECT_ROOT / "data" / "cleaned" / "content" / "text" / f"{key}.pdf.txt"
    return candidate if candidate.exists() else None


def should_append_current_pdf_supplement(
    *,
    source_type: str,
    current_text: str,
    pdf_supplement_text: str,
) -> bool:
    if source_type != "HTML":
        return False
    current_len = len(current_text.strip())
    supplement_len = len(pdf_supplement_text.strip())
    if current_len < 1 or supplement_len < 500:
        return False
    return supplement_len >= max(current_len * 2, current_len + 12000)


def build_unified_text(
    key: str,
    source_type: str,
    current_text: str,
    marker_text: str,
    *,
    current_pdf_supplement_text: str = "",
) -> str:
    parts = [
        f"# Stage1 unified current+Marker document surface: {key}",
        "",
        "<!-- current_clean_text: backward-compatible Stage1 text surface -->",
        current_text.strip(),
    ]
    if source_type == "PDF" and marker_text.strip():
        parts.extend(
            [
                "",
                "---",
                "",
                "<!-- marker_pdf_text: additive frozen Marker structured extraction surface -->",
                marker_text.strip(),
            ]
        )
    if source_type == "HTML" and current_pdf_supplement_text.strip():
        parts.extend(
            [
                "",
                "---",
                "",
                "<!-- current_pdf_clean_text_supplement: additive same-key PDF clean text for HTML completeness fallback -->",
                current_pdf_supplement_text.strip(),
            ]
        )
    return "\n".join(parts).rstrip() + "\n"


def process_one(
    key: str,
    manifest_row: dict[str, str],
    key2txt: dict[str, str],
    key2structure: dict[str, dict[str, str]],
    key2marker: dict[str, dict[str, str]],
    key2cells: dict[str, dict[str, str]],
    out_root: Path,
) -> dict[str, str]:
    rec = {field: "" for field in SUMMARY_FIELDS}
    rec.update(paper_key=key, status="", contract_version=SCRIPT_CONTRACT_VERSION, note="")

    cur_rel = key2txt.get(key) or manifest_row.get("text_path") or manifest_row.get("txt_path") or ""
    if not cur_rel:
        rec.update(status="missing_current_text", note="no key2txt/text_path binding")
        return rec
    cur_text_path = resolve_repo_path(cur_rel)
    if not cur_text_path.exists():
        rec.update(status="missing_current_text", current_text_path=repo_rel(cur_text_path), note="current text path not found")
        return rec

    current_text = cur_text_path.read_text(encoding="utf-8", errors="replace")
    rec.update(
        current_text_path=repo_rel(cur_text_path),
        current_text_sha256=sha256_file(cur_text_path),
        current_text_length=str(len(current_text)),
    )

    cur_struct_rel = (key2structure.get(key) or {}).get("structure_path") or manifest_row.get("structure_path") or ""
    cur_struct, cur_struct_path_rel, cur_struct_sha = load_json_if_exists(cur_struct_rel)
    if cur_struct_path_rel:
        rec.update(
            current_structure_path=cur_struct_path_rel,
            current_structure_sha256=cur_struct_sha,
            current_block_count=str(len(cur_struct.get("blocks") or [])),
        )
    source_type = infer_source_type({**manifest_row, "text_path": rec["current_text_path"]}, cur_struct)
    rec["source_type"] = source_type
    manifest_table_dir = str(manifest_row.get("table_dir") or "").strip()
    manifest_table_available = str(manifest_row.get("table_available") or "").strip()
    if manifest_table_dir:
        rec.update(
            table_dir=manifest_table_dir,
            table_available=manifest_table_available or "yes",
        )

    marker_row = key2marker.get(key) or {}
    marker_text = ""
    marker_struct: dict[str, Any] = {}
    marker_text_rel = marker_row.get("marker_text_path") or ""
    marker_struct_rel = marker_row.get("marker_structure_path") or ""
    if marker_text_rel:
        mp = resolve_repo_path(marker_text_rel)
        if mp.exists():
            marker_text = mp.read_text(encoding="utf-8", errors="replace")
            rec.update(marker_text_path=repo_rel(mp), marker_text_sha256=sha256_file(mp), marker_text_length=str(len(marker_text)))
    if marker_struct_rel:
        marker_struct, marker_struct_path_rel, marker_struct_sha = load_json_if_exists(marker_struct_rel)
        if marker_struct_path_rel:
            rec.update(
                marker_structure_path=marker_struct_path_rel,
                marker_structure_sha256=marker_struct_sha,
                marker_block_count=str(len(marker_struct.get("blocks") or [])),
            )

    current_pdf_supplement_text = ""
    pdf_supplement_path = html_pdf_sibling_supplement_path(cur_text_path)
    if pdf_supplement_path is None and source_type == "HTML":
        pdf_supplement_path = canonical_pdf_clean_text_supplement_path(key)
    if pdf_supplement_path is not None and pdf_supplement_path.resolve() != cur_text_path.resolve():
        candidate_text = pdf_supplement_path.read_text(encoding="utf-8", errors="replace")
        if should_append_current_pdf_supplement(
            source_type=source_type,
            current_text=current_text,
            pdf_supplement_text=candidate_text,
        ):
            current_pdf_supplement_text = candidate_text

    paper_out = out_root / key
    paper_out.mkdir(parents=True, exist_ok=True)

    cell_row = key2cells.get(key) or {}
    sidecar_path = (
        cell_row.get("stage1_table_cell_sidecar_path")
        or manifest_row.get("stage1_table_cell_sidecar_path")
        or ""
    ).strip()
    sidecar_available = (
        cell_row.get("stage1_table_cell_sidecar_available")
        or manifest_row.get("stage1_table_cell_sidecar_available")
        or ""
    ).strip()
    if sidecar_path:
        rec.update(
            stage1_table_cell_sidecar_path=sidecar_path,
            stage1_table_cell_sidecar_available=sidecar_available or "yes",
            table_cell_count=cell_row.get("cell_count", "") or manifest_row.get("table_cell_count", ""),
            table_count=cell_row.get("table_count", "") or manifest_row.get("table_count", ""),
        )
    else:
        rec.update(stage1_table_cell_sidecar_available="no")

    unified_text_path = paper_out / "unified_clean_text_v1.md"
    unified_structure_path = paper_out / "unified_structure_v1.json"

    unified_text = build_unified_text(
        key,
        source_type,
        current_text,
        marker_text,
        current_pdf_supplement_text=current_pdf_supplement_text,
    )
    current_blocks = normalize_current_blocks(cur_struct.get("blocks") or [], source_type)
    marker_blocks = normalize_marker_blocks(marker_struct.get("blocks") or [], start_order=len(current_blocks) + 1) if source_type == "PDF" else []
    unified_blocks = current_blocks + marker_blocks
    sidecar_tables: list[dict[str, Any]] = []
    if sidecar_path and not rec["table_dir"] and not (cell_row.get("table_dir") or "").strip():
        sidecar_tables, sidecar_table_dir, sidecar_table_available = extract_table_assets_from_stage1_sidecar(
            paper_key=key,
            sidecar_path=sidecar_path,
            paper_out=paper_out,
        )
        if sidecar_tables:
            rec.update(table_dir=sidecar_table_dir, table_available=sidecar_table_available)
    marker_tables, marker_sidecar_rows, marker_table_dir, marker_table_available, marker_cell_count, marker_table_count = extract_marker_table_assets_from_blocks(
        paper_key=key,
        marker_blocks=marker_struct.get("blocks") or [],
        paper_out=paper_out,
    ) if source_type == "PDF" and not sidecar_path else ([], [], "", "no", "0", "0")
    if marker_tables and not sidecar_path and not rec["table_dir"]:
        rec.update(
            table_dir=marker_table_dir,
            table_available=marker_table_available,
            stage1_table_cell_sidecar_path=repo_rel(paper_out / "stage1_table_cells_v1.jsonl"),
            stage1_table_cell_sidecar_available="yes",
            table_cell_count=marker_cell_count,
            table_count=marker_table_count,
        )
    elif sidecar_path:
        rec.update(
            table_dir=rec["table_dir"] or cell_row.get("table_dir", ""),
            table_available=rec["table_available"] or (cell_row.get("table_available", "yes") if rec["stage1_table_cell_sidecar_available"] == "yes" else "no"),
        )
    elif not rec["table_dir"]:
        rec.update(table_available="no")
    unified_tables = list(cur_struct.get("tables") or [])
    if sidecar_tables:
        unified_tables.extend(
            {
                **item,
                "source": "stage1_table_cell_sidecar_projection",
            }
            for item in sidecar_tables
        )
    if marker_tables:
        unified_tables.extend(
            {
                **item,
                "source": "marker_pdf_table_block_promotion",
            }
            for item in marker_tables
        )

    unified_payload = {
        "doc_key": key,
        "paper_key": key,
        "source_type": source_type,
        "parser": "stage1_current_plus_marker_unified",
        "contract_version": SCRIPT_CONTRACT_VERSION,
        "text_path": repo_rel(unified_text_path),
        "structure_path": repo_rel(unified_structure_path),
        "text_sha256": sha256_text(unified_text),
        "sources": {
            "current": {
                "text_path": rec["current_text_path"],
                "structure_path": rec["current_structure_path"],
                "text_sha256": rec["current_text_sha256"],
                "structure_sha256": rec["current_structure_sha256"],
                "source_type": source_type,
                "block_count": int(rec["current_block_count"] or 0),
            },
            "marker_pdf": {
                "available": bool(rec["marker_text_path"] or rec["marker_structure_path"]),
                "text_path": rec["marker_text_path"],
                "structure_path": rec["marker_structure_path"],
                "text_sha256": rec["marker_text_sha256"],
                "structure_sha256": rec["marker_structure_sha256"],
                "block_count": int(rec["marker_block_count"] or 0),
                "reuse_contract": "frozen Marker output consumed only when key2marker_pdf_v1.tsv already binds it",
            },
            "current_pdf_clean_text_supplement": {
                "available": bool(current_pdf_supplement_text),
                "text_path": repo_rel(pdf_supplement_path) if current_pdf_supplement_text and pdf_supplement_path is not None else "",
                "text_length": len(current_pdf_supplement_text),
                "selection_rule": "same_key_pdf_clean_text_appended_when_html_clean_text_is_much_shorter",
            },
            "table_cells": {
                "available": rec["stage1_table_cell_sidecar_available"] == "yes",
                "sidecar_path": rec["stage1_table_cell_sidecar_path"],
                "cell_count": rec["table_cell_count"],
                "table_count": rec["table_count"],
            },
        },
        "blocks": unified_blocks,
        "tables": unified_tables,
        "stage1_table_cell_sidecar_path": rec["stage1_table_cell_sidecar_path"],
        "stage1_table_cell_sidecar_available": rec["stage1_table_cell_sidecar_available"],
        "table_dir": rec["table_dir"],
        "table_available": rec["table_available"],
        "metadata": {
            "policy": "single additive Stage1 interface for current HTML/PDF plus frozen Marker PDF where available",
            "current_text_is_compatibility_base": True,
            "marker_rerun_performed": False,
            "no_semantic_inference": True,
            "stage2_stage5_not_run": True,
            "block_projection": "top-level blocks list is normalized for Stage2 structure-sidecar matching; source parser is retained per block",
            "marker_table_block_promotion": "Marker Table/TableCell blocks are structurally promoted to table_dir CSV and cell sidecar assets when no stronger Stage1 table sidecar is already bound",
            "html_pdf_clean_text_supplement": "HTML remains the selected current binding, but same-key PDF clean text is appended when the HTML clean text is much shorter and would otherwise under-cover the source body",
        },
    }

    unified_text_path.write_text(unified_text, encoding="utf-8")
    unified_structure_path.write_text(json.dumps(unified_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rec.update(
        status="unified",
        table_authority_inherited=manifest_row.get("table_authority_inherited", "no"),
        table_authority_inheritance_source=manifest_row.get("table_authority_inheritance_source", ""),
        note=(
            "OK; HTML includes same-key PDF clean-text supplement"
            if current_pdf_supplement_text
            else ("OK" if not (source_type == "PDF" and not marker_blocks) else "OK; PDF has no frozen Marker block attachment")
        ),
        unified_text_path=repo_rel(unified_text_path),
        unified_structure_path=repo_rel(unified_structure_path),
        unified_text_sha256=sha256_file(unified_text_path),
        unified_structure_sha256=sha256_file(unified_structure_path),
        unified_text_length=str(len(unified_text)),
        unified_block_count=str(len(unified_blocks)),
    )
    return rec


def write_outputs(
    rows: list[dict[str, str]],
    source_manifest_rows: dict[str, dict[str, str]],
    summary_out: Path,
    key2unified_out: Path,
    manifest_out: Path,
) -> None:
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    merged: dict[str, dict[str, str]] = {}
    if key2unified_out.exists():
        with key2unified_out.open("r", encoding="utf-8", errors="replace", newline="") as f:
            for old in csv.DictReader(f, delimiter="\t"):
                key = (old.get("paper_key") or "").strip()
                if key:
                    merged[key] = {k: old.get(k, "") for k in KEY2_FIELDS}
    key2_current_rows: dict[str, dict[str, str]] = {}
    for r in rows:
        if r.get("status") == "unified":
            key2_row = {
                "paper_key": r["paper_key"],
                "text_path": r["unified_text_path"],
                "structure_path": r["unified_structure_path"],
                "text_source_type": r["source_type"].lower(),
                "text_available": "yes",
                "structure_available": "yes",
                "table_dir": r["table_dir"],
                "table_available": r["table_available"],
                "stage1_table_cell_sidecar_path": r["stage1_table_cell_sidecar_path"],
                "stage1_table_cell_sidecar_available": r["stage1_table_cell_sidecar_available"],
                "table_cell_count": r["table_cell_count"],
                "table_count": r["table_count"],
                "table_authority_inherited": r["table_authority_inherited"],
                "table_authority_inheritance_source": r["table_authority_inheritance_source"],
                "unified_text_sha256": r["unified_text_sha256"],
                "unified_structure_sha256": r["unified_structure_sha256"],
                "status": r["status"],
            }
            merged[r["paper_key"]] = key2_row
            key2_current_rows[r["paper_key"]] = key2_row
    with key2unified_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEY2_FIELDS, delimiter="\t")
        w.writeheader()
        for key in sorted(merged):
            w.writerow(merged[key])

    manifest_rows: list[dict[str, str]] = []
    base_fields: list[str] = []
    for key in sorted(key2_current_rows):
        source_row = dict(source_manifest_rows.get(key, {}))
        if not base_fields:
            base_fields = list(source_row.keys())
        key2_row = key2_current_rows[key]
        source_row["paper_key"] = key
        source_row["key"] = source_row.get("key") or key
        source_row.update(
            {
                "text_path": key2_row["text_path"],
                "txt_path": key2_row["text_path"],
                "text_source_type": key2_row["text_source_type"],
                "text_available": key2_row["text_available"],
                "structure_path": key2_row["structure_path"],
                "structure_available": key2_row["structure_available"],
                "table_dir": key2_row["table_dir"],
                "table_available": key2_row["table_available"],
                "stage1_table_cell_sidecar_path": key2_row["stage1_table_cell_sidecar_path"],
                "stage1_table_cell_sidecar_available": key2_row["stage1_table_cell_sidecar_available"],
                "table_authority_inherited": key2_row["table_authority_inherited"],
                "table_authority_inheritance_source": key2_row["table_authority_inheritance_source"],
                "stage1_unified_contract_version": SCRIPT_CONTRACT_VERSION,
                "stage1_unified_source": "current_clean_text_plus_frozen_marker_pdf_when_available",
            }
        )
        manifest_rows.append(source_row)
    manifest_fields = list(dict.fromkeys(base_fields + [
        "paper_key",
        "key",
        "text_path",
        "txt_path",
        "text_source_type",
        "text_available",
        "structure_path",
        "structure_available",
        "table_dir",
        "table_available",
        "stage1_table_cell_sidecar_path",
        "stage1_table_cell_sidecar_available",
        "table_authority_inherited",
        "table_authority_inheritance_source",
        "stage1_unified_contract_version",
        "stage1_unified_source",
    ]))
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build unified governed Stage1 current+Marker document artifacts for PDF and HTML.")
    ap.add_argument("--manifest", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "manifest_current.tsv")
    ap.add_argument("--scope-keys", type=Path, default=None)
    ap.add_argument("--key2txt", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "key2txt.tsv")
    ap.add_argument("--key2structure", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "key2structure.tsv")
    ap.add_argument("--key2marker", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "key2marker_pdf_v1.tsv")
    ap.add_argument("--table-cell-manifest", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "stage1_table_cells_manifest_v1.tsv")
    ap.add_argument(
        "--table-authority-manifest",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional explicit upstream Stage1 authority manifest. For each key, "
            "table_dir and stage1_table_cell_sidecar_path are inherited only when "
            "the primary manifest row lacks that authority."
        ),
    )
    ap.add_argument("--content-dir", type=Path, default=paths.DATA_CLEANED_CONTENT_DIR)
    ap.add_argument("--summary-out", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "stage1_unified_current_marker_summary_v1.tsv")
    ap.add_argument("--key2unified-out", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "key2stage1_unified_current_marker_v1.tsv")
    ap.add_argument("--manifest-out", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "stage1_unified_current_marker_manifest_v1.tsv")
    args = ap.parse_args()

    scope = read_scope(args.scope_keys)
    manifest = load_manifest(args.manifest, scope)
    table_authority_rows = load_table_authority_rows(args.table_authority_manifest)
    if table_authority_rows:
        manifest = {
            key: merge_table_authority_row(row, table_authority_rows.get(key))
            for key, row in manifest.items()
        }
    key2txt = load_key2txt(args.key2txt)
    key2structure = load_key2structure(args.key2structure)
    key2marker = load_tsv_by_key(args.key2marker, key_field="paper_key")
    key2cells = load_tsv_by_key(args.table_cell_manifest, key_field="paper_key")
    out_root = args.content_dir / "stage1_unified_current_marker_v1"

    rows: list[dict[str, str]] = []
    for key in sorted(manifest):
        rows.append(process_one(key, manifest[key], key2txt, key2structure, key2marker, key2cells, out_root))
    write_outputs(rows, manifest, args.summary_out, args.key2unified_out, args.manifest_out)

    unified = sum(1 for r in rows if r["status"] == "unified")
    pdf_with_marker = sum(1 for r in rows if r["status"] == "unified" and r["source_type"] == "PDF" and r["marker_structure_path"])
    html_with_cells = sum(1 for r in rows if r["status"] == "unified" and r["source_type"] == "HTML" and r["stage1_table_cell_sidecar_available"] == "yes")
    print(f"[OK] unified summary -> {args.summary_out}")
    print(f"[OK] key2unified -> {args.key2unified_out}")
    print(f"[OK] unified manifest -> {args.manifest_out}")
    print(f"[INFO] unified={unified} pdf_with_marker={pdf_with_marker} html_with_table_cells={html_with_cells} total={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
