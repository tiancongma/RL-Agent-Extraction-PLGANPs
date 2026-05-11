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
- Current clean text remains the compatibility base; Marker text is additive for
  PDF records only when frozen Marker output exists.
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
    "table_cell_count",
    "table_count",
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
    "stage1_table_cell_sidecar_path",
    "stage1_table_cell_sidecar_available",
    "table_cell_count",
    "table_count",
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
                    out[key] = val
        else:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0] and parts[1]:
                    out[parts[0]] = parts[1]
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


def load_json_if_exists(rel_or_abs: str) -> tuple[dict[str, Any], str, str]:
    if not rel_or_abs:
        return {}, "", ""
    p = resolve_repo_path(rel_or_abs)
    if not p.exists():
        return {}, "", ""
    return json.loads(p.read_text(encoding="utf-8", errors="replace")), repo_rel(p), sha256_file(p)


def infer_source_type(row: dict[str, str], current_structure: dict[str, Any]) -> str:
    for val in [row.get("text_source_type", ""), current_structure.get("source_type", "")]:
        if str(val).strip():
            return str(val).strip().upper()
    txt = row.get("text_path") or row.get("txt_path") or ""
    if ".html" in txt.lower():
        return "HTML"
    if ".pdf" in txt.lower():
        return "PDF"
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


def build_unified_text(key: str, source_type: str, current_text: str, marker_text: str) -> str:
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

    cell_row = key2cells.get(key) or {}
    sidecar_path = (cell_row.get("stage1_table_cell_sidecar_path") or "").strip()
    sidecar_available = (cell_row.get("stage1_table_cell_sidecar_available") or "").strip()
    if sidecar_path:
        rec.update(
            stage1_table_cell_sidecar_path=sidecar_path,
            stage1_table_cell_sidecar_available=sidecar_available or "yes",
            table_cell_count=cell_row.get("cell_count", ""),
            table_count=cell_row.get("table_count", ""),
        )
    else:
        rec.update(stage1_table_cell_sidecar_available="no")

    paper_out = out_root / key
    paper_out.mkdir(parents=True, exist_ok=True)
    unified_text_path = paper_out / "unified_clean_text_v1.md"
    unified_structure_path = paper_out / "unified_structure_v1.json"

    unified_text = build_unified_text(key, source_type, current_text, marker_text)
    current_blocks = normalize_current_blocks(cur_struct.get("blocks") or [], source_type)
    marker_blocks = normalize_marker_blocks(marker_struct.get("blocks") or [], start_order=len(current_blocks) + 1) if source_type == "PDF" else []
    unified_blocks = current_blocks + marker_blocks

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
            "table_cells": {
                "available": rec["stage1_table_cell_sidecar_available"] == "yes",
                "sidecar_path": rec["stage1_table_cell_sidecar_path"],
                "cell_count": rec["table_cell_count"],
                "table_count": rec["table_count"],
            },
        },
        "blocks": unified_blocks,
        "tables": cur_struct.get("tables") or [],
        "stage1_table_cell_sidecar_path": rec["stage1_table_cell_sidecar_path"],
        "stage1_table_cell_sidecar_available": rec["stage1_table_cell_sidecar_available"],
        "metadata": {
            "policy": "single additive Stage1 interface for current HTML/PDF plus frozen Marker PDF where available",
            "current_text_is_compatibility_base": True,
            "marker_rerun_performed": False,
            "no_semantic_inference": True,
            "stage2_stage5_not_run": True,
            "block_projection": "top-level blocks list is normalized for Stage2 structure-sidecar matching; source parser is retained per block",
        },
    }

    unified_text_path.write_text(unified_text, encoding="utf-8")
    unified_structure_path.write_text(json.dumps(unified_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rec.update(
        status="unified",
        note="OK" if not (source_type == "PDF" and not marker_blocks) else "OK; PDF has no frozen Marker block attachment",
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
                "stage1_table_cell_sidecar_path": r["stage1_table_cell_sidecar_path"],
                "stage1_table_cell_sidecar_available": r["stage1_table_cell_sidecar_available"],
                "table_cell_count": r["table_cell_count"],
                "table_count": r["table_count"],
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
                "stage1_table_cell_sidecar_path": key2_row["stage1_table_cell_sidecar_path"],
                "stage1_table_cell_sidecar_available": key2_row["stage1_table_cell_sidecar_available"],
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
        "stage1_table_cell_sidecar_path",
        "stage1_table_cell_sidecar_available",
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
    ap.add_argument("--content-dir", type=Path, default=paths.DATA_CLEANED_CONTENT_DIR)
    ap.add_argument("--summary-out", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "stage1_unified_current_marker_summary_v1.tsv")
    ap.add_argument("--key2unified-out", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "key2stage1_unified_current_marker_v1.tsv")
    ap.add_argument("--manifest-out", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "stage1_unified_current_marker_manifest_v1.tsv")
    args = ap.parse_args()

    scope = read_scope(args.scope_keys)
    manifest = load_manifest(args.manifest, scope)
    key2txt = load_key2txt(args.key2txt)
    key2structure = load_tsv_by_key(args.key2structure, key_field="key")
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
