#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persist diagnostic-only S2-1 paper scope context artifacts.

This helper records the per-paper Stage2 input boundary from an explicit scope
TSV plus the single canonical Stage1 manifest. It does not run semantic
extraction, does not select evidence, does not call an LLM, and does not create
a competing manifest authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
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

SCHEMA_ID = "paper_scope_context_v1"
AUDIT_FIELDS = [
    "paper_key",
    "context_path",
    "canonical_manifest_path",
    "canonical_manifest_sha256",
    "scope_tsv_path",
    "selection_reason",
    "clean_text_path",
    "clean_text_exists",
    "text_source_type",
    "pdf_path",
    "pdf_exists",
    "html_path",
    "html_exists",
    "table_asset_root",
    "table_asset_root_exists",
    "table_asset_refs",
    "table_asset_ref_count",
    "table_available_status",
]


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def resolve_project_path(value: Path | str) -> Path:
    p = value if isinstance(value, Path) else Path(value)
    return p if p.is_absolute() else (paths.PROJECT_ROOT / p).resolve()


def display_path(p: Path | str | None) -> str:
    if not p:
        return ""
    pp = p if isinstance(p, Path) else Path(str(p).replace("\\", "/"))
    try:
        if pp.is_absolute():
            return str(pp)
        return str(pp)
    except Exception:
        return str(p)


def resolve_optional_path(value: Any) -> Path | None:
    text = normalize_text(value).replace("\\", "/")
    if not text:
        return None
    p = Path(text)
    return p if p.is_absolute() else (paths.PROJECT_ROOT / p).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in AUDIT_FIELDS})


def row_key(row: dict[str, str]) -> str:
    return normalize_text(row.get("paper_key") or row.get("key") or row.get("zotero_key") or row.get("paper_id"))


def first_present(row: dict[str, str], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = normalize_text(row.get(field))
        if value:
            return value
    return ""


def split_refs(value: str) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    parts: list[str] = []
    for chunk in text.replace(";", "|").replace(",", "|").split("|"):
        item = chunk.strip()
        if item:
            parts.append(item)
    return parts


def resolve_table_root(scope_row: dict[str, str], manifest_row: dict[str, str]) -> str:
    return first_present(scope_row, ("table_asset_root", "table_dir", "tables_root")) or first_present(
        manifest_row, ("table_asset_root", "table_dir", "tables_root")
    )


def resolve_table_refs(scope_row: dict[str, str], manifest_row: dict[str, str], table_root: str) -> list[str]:
    raw = first_present(scope_row, ("table_asset_refs", "table_manifest_path", "tables_manifest_path", "table_files")) or first_present(
        manifest_row, ("table_asset_refs", "table_manifest_path", "tables_manifest_path", "table_files")
    )
    refs = split_refs(raw)
    root_path = resolve_optional_path(table_root)
    if not refs and root_path and root_path.exists():
        manifest_candidate = root_path / "tables_manifest.json"
        if manifest_candidate.exists():
            try:
                refs.append(str(manifest_candidate.relative_to(paths.PROJECT_ROOT)))
            except ValueError:
                refs.append(str(manifest_candidate))
    return refs


def infer_table_root_from_text(text_path: str, paper_key: str) -> str:
    text = resolve_optional_path(text_path)
    if not text:
        return ""
    candidates = [
        text.parent.parent / "tables" / paper_key,
        paths.PROJECT_ROOT / "data" / "cleaned" / "goren_2025" / "tables" / paper_key,
        paths.PROJECT_ROOT / "data" / "cleaned" / "content_goren_2025" / "tables" / paper_key,
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return str(candidate.relative_to(paths.PROJECT_ROOT))
            except ValueError:
                return str(candidate)
    return ""


def build_context_for_row(
    *,
    scope_row: dict[str, str],
    manifest_row: dict[str, str],
    canonical_manifest_path: Path,
    canonical_manifest_sha256: str,
    scope_tsv_path: Path,
    run_id: str,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    paper_key = row_key(scope_row) or row_key(manifest_row)
    if not paper_key:
        raise RuntimeError("scope row lacks paper_key/key/zotero_key/paper_id")

    text_path = first_present(scope_row, ("text_path", "clean_text_path", "source_text_path")) or first_present(
        manifest_row, ("text_path", "clean_text_path", "source_text_path")
    )
    text_source_type = first_present(scope_row, ("text_source_type",)) or first_present(manifest_row, ("text_source_type",))
    pdf_path = first_present(scope_row, ("pdf", "pdf_path")) or first_present(manifest_row, ("pdf", "pdf_path"))
    html_path = first_present(scope_row, ("html", "html_path")) or first_present(manifest_row, ("html", "html_path"))
    table_root = resolve_table_root(scope_row, manifest_row)
    if not table_root:
        table_root = infer_table_root_from_text(text_path, paper_key)
    table_refs = resolve_table_refs(scope_row, manifest_row, table_root)

    text_abs = resolve_optional_path(text_path)
    pdf_abs = resolve_optional_path(pdf_path)
    html_abs = resolve_optional_path(html_path)
    table_root_abs = resolve_optional_path(table_root)
    ref_paths = [resolve_optional_path(ref) for ref in table_refs]
    available_refs = [p for p in ref_paths if p and p.exists()]
    table_available_status = "available" if (table_root_abs and table_root_abs.exists()) or available_refs else "explicit_absent"

    rel_context_path = Path("semantic_stage2_objects") / "scope_context" / paper_key / "paper_scope_context_v1.json"
    context_path = out_dir / rel_context_path
    context: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "diagnostic_only": True,
        "benchmark_valid": False,
        "paper_key": paper_key,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canonical_manifest": {
            "path": str(canonical_manifest_path),
            "sha256": canonical_manifest_sha256,
            "authority_role": "single_canonical_manifest",
        },
        "run_scope": {
            "scope_tsv_path": str(scope_tsv_path),
            "selection_reason": first_present(scope_row, ("selection_reason", "pilot_reason")),
            "scope_row_key_fields": {
                "key": normalize_text(scope_row.get("key")),
                "paper_key": normalize_text(scope_row.get("paper_key")),
                "zotero_key": normalize_text(scope_row.get("zotero_key")),
                "paper_id": normalize_text(scope_row.get("paper_id")),
            },
        },
        "paper_metadata": {
            "title": first_present(scope_row, ("title",)) or first_present(manifest_row, ("title",)),
            "doi": first_present(scope_row, ("doi", "normalized_doi")) or first_present(manifest_row, ("doi", "normalized_doi")),
            "year": first_present(scope_row, ("year",)) or first_present(manifest_row, ("year",)),
            "dataset_id": first_present(scope_row, ("dataset_id",)) or first_present(manifest_row, ("dataset_id",)),
            "split_tag": first_present(scope_row, ("split_tag",)) or first_present(manifest_row, ("split_tag",)),
            "benchmark_tag": first_present(scope_row, ("benchmark_tag",)) or first_present(manifest_row, ("benchmark_tag",)),
            "source_collection": first_present(scope_row, ("source_collection", "collection", "library_id", "dataset_source"))
            or first_present(manifest_row, ("source_collection", "collection", "library_id", "dataset_source")),
            "source_record_path": first_present(scope_row, ("source_record_path", "raw_source_path", "raw_jsonl_path"))
            or first_present(manifest_row, ("source_record_path", "raw_source_path", "raw_jsonl_path")),
        },
        "clean_text": {
            "path": text_path,
            "resolved_path": str(text_abs) if text_abs else "",
            "source_type": text_source_type,
            "exists": bool(text_abs and text_abs.exists()),
        },
        "source_assets": {
            "pdf_path": pdf_path,
            "pdf_resolved_path": str(pdf_abs) if pdf_abs else "",
            "pdf_exists": bool(pdf_abs and pdf_abs.exists()),
            "html_path": html_path,
            "html_resolved_path": str(html_abs) if html_abs else "",
            "html_exists": bool(html_abs and html_abs.exists()),
        },
        "table_assets": {
            "table_asset_root": table_root,
            "table_asset_root_resolved_path": str(table_root_abs) if table_root_abs else "",
            "table_asset_root_exists": bool(table_root_abs and table_root_abs.exists()),
            "table_asset_refs": table_refs,
            "table_asset_ref_resolved_paths": [str(p) if p else "" for p in ref_paths],
            "table_asset_ref_exists": [bool(p and p.exists()) for p in ref_paths],
            "availability_status": table_available_status,
        },
        "governance": {
            "s2_substep": "S2-1 Scope resolution",
            "does_rank_or_select_evidence": False,
            "does_semantically_interpret_evidence": False,
            "live_llm_call": False,
            "notes": "Diagnostic-only per-paper input boundary; downstream evidence selection remains S2-2.",
        },
    }
    audit_row = {
        "paper_key": paper_key,
        "context_path": str(context_path),
        "canonical_manifest_path": str(canonical_manifest_path),
        "canonical_manifest_sha256": canonical_manifest_sha256,
        "scope_tsv_path": str(scope_tsv_path),
        "selection_reason": context["run_scope"]["selection_reason"],
        "clean_text_path": text_path,
        "clean_text_exists": "yes" if context["clean_text"]["exists"] else "no",
        "text_source_type": text_source_type,
        "pdf_path": pdf_path,
        "pdf_exists": "yes" if context["source_assets"]["pdf_exists"] else "no",
        "html_path": html_path,
        "html_exists": "yes" if context["source_assets"]["html_exists"] else "no",
        "table_asset_root": table_root,
        "table_asset_root_exists": "yes" if context["table_assets"]["table_asset_root_exists"] else "no",
        "table_asset_refs": "|".join(table_refs),
        "table_asset_ref_count": str(len(table_refs)),
        "table_available_status": table_available_status,
    }
    return context, audit_row


def build_paper_scope_contexts(
    *,
    canonical_manifest: Path,
    scope_tsv: Path,
    out_dir: Path,
    run_id: str,
    paper_keys: list[str] | None = None,
) -> dict[str, Any]:
    canonical_manifest = resolve_project_path(canonical_manifest)
    scope_tsv = resolve_project_path(scope_tsv)
    out_dir = resolve_project_path(out_dir)
    if not canonical_manifest.exists():
        raise FileNotFoundError(f"canonical manifest not found: {canonical_manifest}")
    if not scope_tsv.exists():
        raise FileNotFoundError(f"scope TSV not found: {scope_tsv}")
    if not run_id:
        raise ValueError("run_id is required")

    manifest_rows = read_tsv(canonical_manifest)
    scope_rows = read_tsv(scope_tsv)
    manifest_by_key = {row_key(row): row for row in manifest_rows if row_key(row)}
    selected_keys = {normalize_text(k) for k in (paper_keys or []) if normalize_text(k)}
    if selected_keys:
        scope_rows = [row for row in scope_rows if row_key(row) in selected_keys]
    if not scope_rows:
        raise RuntimeError("scope selection resolved zero rows")

    manifest_hash = sha256_file(canonical_manifest)
    audit_rows: list[dict[str, str]] = []
    missing_manifest: list[str] = []
    for scope_row in scope_rows:
        key = row_key(scope_row)
        manifest_row = manifest_by_key.get(key, {})
        if not manifest_row:
            missing_manifest.append(key)
        context, audit_row = build_context_for_row(
            scope_row=scope_row,
            manifest_row=manifest_row,
            canonical_manifest_path=canonical_manifest,
            canonical_manifest_sha256=manifest_hash,
            scope_tsv_path=scope_tsv,
            run_id=run_id,
            out_dir=out_dir,
        )
        context_path = Path(audit_row["context_path"])
        context_path.parent.mkdir(parents=True, exist_ok=True)
        context_path.write_text(json.dumps(context, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        audit_rows.append(audit_row)

    audit_path = out_dir / "analysis" / "paper_scope_context_audit_v1.tsv"
    write_tsv(audit_path, audit_rows)
    run_context_path = out_dir / "RUN_CONTEXT.md"
    run_context_path.parent.mkdir(parents=True, exist_ok=True)
    run_context_path.write_text(
        "# RUN_CONTEXT\n\n"
        "- run_type: diagnostic-only S2-1 paper scope context materialization\n"
        "- benchmark_valid: no\n"
        f"- run_id: {run_id}\n"
        f"- canonical_manifest: {canonical_manifest}\n"
        f"- canonical_manifest_sha256: {manifest_hash}\n"
        f"- scope_tsv: {scope_tsv}\n"
        f"- generated_by: src/stage2_sampling_labels/build_paper_scope_context_v1.py\n"
        f"- paper_count: {len(audit_rows)}\n"
        f"- audit_tsv: {audit_path}\n",
        encoding="utf-8",
    )
    return {
        "schema": "paper_scope_context_run_summary_v1",
        "diagnostic_only": True,
        "benchmark_valid": False,
        "run_id": run_id,
        "out_dir": str(out_dir),
        "canonical_manifest": str(canonical_manifest),
        "canonical_manifest_sha256": manifest_hash,
        "scope_tsv": str(scope_tsv),
        "paper_count": len(audit_rows),
        "paper_keys": [row["paper_key"] for row in audit_rows],
        "missing_manifest_paper_keys": missing_manifest,
        "audit_tsv": str(audit_path),
        "run_context_md": str(run_context_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", type=Path, required=True, help="Explicit single canonical manifest path.")
    parser.add_argument("--scope-tsv", type=Path, required=True, help="Explicit run scope TSV path.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Diagnostic run output directory.")
    parser.add_argument("--run-id", required=True, help="Explicit diagnostic run id recorded in contexts.")
    parser.add_argument("--paper-key", action="append", dest="paper_keys", default=[], help="Optional bounded paper-key filter.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = build_paper_scope_contexts(
        canonical_manifest=args.canonical_manifest,
        scope_tsv=args.scope_tsv,
        out_dir=args.out_dir,
        run_id=args.run_id,
        paper_keys=args.paper_keys,
    )
    print("diagnostic_only=true")
    print("benchmark_valid=false")
    print(f"canonical_manifest={summary['canonical_manifest']}")
    print(f"canonical_manifest_sha256={summary['canonical_manifest_sha256']}")
    print(f"scope_tsv={summary['scope_tsv']}")
    print(f"out_dir={summary['out_dir']}")
    print(f"paper_count={summary['paper_count']}")
    print("paper_keys=" + ",".join(summary["paper_keys"]))
    print(f"audit_tsv={summary['audit_tsv']}")
    print(f"run_context={summary['run_context_md']}")
    if summary["missing_manifest_paper_keys"]:
        print("missing_manifest_paper_keys=" + ",".join(summary["missing_manifest_paper_keys"]))


if __name__ == "__main__":
    main()
