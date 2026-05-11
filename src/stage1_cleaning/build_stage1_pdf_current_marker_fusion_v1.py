#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage1_pdf_current_marker_fusion_v1.py

Purpose
- Governed Stage1 current+Marker PDF fusion helper.
- Fuse the existing current PDF clean-text/structure surface with frozen Marker
  PDF outputs without rerunning Marker and without changing Stage2/benchmark.

Contract
- Inputs are governed Stage1 indexes:
    data/cleaned/index/manifest_current.tsv
    data/cleaned/index/key2txt.tsv
    data/cleaned/index/key2structure.tsv
    data/cleaned/index/key2marker_pdf_v1.tsv
- Outputs are additive governed Stage1 artifacts:
    data/cleaned/content/pdf_current_marker_fusion/<paper_key>/fused_clean_text_v1.md
    data/cleaned/content/pdf_current_marker_fusion/<paper_key>/fused_structure_v1.json
    data/cleaned/index/key2stage1_pdf_fusion_v1.tsv
    data/cleaned/index/stage1_pdf_current_marker_fusion_summary_v1.tsv

Principle
- Current PDF text remains compatibility text authority.
- Marker contributes additive heading/block/table/geometry structure and extra
  text visibility when present.
- No PLGA semantic inference, no Stage2/Stage5 execution, no benchmark claim.
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

SCRIPT_CONTRACT_VERSION = "stage1_pdf_current_marker_fusion_v1"


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


def resolve_repo_path(value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return paths.PROJECT_ROOT / p


def read_scope(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    keys: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames and any(k in reader.fieldnames for k in ["paper_key", "key", "zotero_key"]):
            for row in reader:
                key = (row.get("paper_key") or row.get("key") or row.get("zotero_key") or "").strip()
                if key:
                    keys.add(key)
            return keys
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#") and s not in {"paper_key", "key"}:
                keys.add(s.split("\t")[0].split(",")[0].strip())
    return keys


def load_manifest_pdf_keys(path: Path, scope: set[str] | None) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row.get("paper_key") or row.get("key") or row.get("zotero_key") or "").strip()
            if not key or (scope is not None and key not in scope):
                continue
            if (row.get("text_source_type") or "").lower() == "pdf" or (row.get("pdf") or "").strip():
                out[key] = row
    return out


def load_key2txt(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        sample = f.readline()
        if sample.startswith("key\t"):
            # headered defensive support
            f.seek(0)
            for row in csv.DictReader(f, delimiter="\t"):
                key = (row.get("key") or row.get("paper_key") or "").strip()
                val = (row.get("txt_path") or row.get("text_path") or "").strip()
                if key and val:
                    out[key] = val
        else:
            if sample.strip():
                parts = sample.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    out[parts[0]] = parts[1]
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0] and parts[1]:
                    out[parts[0]] = parts[1]
    return out


def load_tsv_by_key(path: Path, key_field: str = "key") -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row.get(key_field) or row.get("paper_key") or row.get("key") or "").strip()
            if key:
                out[key] = row
    return out


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def build_fused_text(current_text: str, marker_text: str, key: str) -> str:
    parts = [
        f"# Stage1 current+Marker PDF fusion: {key}",
        "",
        "<!-- current_pdf_text: backward-compatible Stage1 clean text surface -->",
        current_text.strip(),
    ]
    if marker_text.strip():
        parts.extend([
            "",
            "---",
            "",
            "<!-- marker_pdf_text: additive frozen Marker structured extraction surface -->",
            marker_text.strip(),
        ])
    return "\n".join(parts).rstrip() + "\n"


def process_one(key: str, manifest_row: dict[str, str], key2txt: dict[str, str], key2structure: dict[str, dict[str, str]], key2marker: dict[str, dict[str, str]], out_root: Path) -> dict[str, str]:
    rec = {
        "paper_key": key,
        "status": "",
        "note": "",
        "current_text_path": "",
        "current_structure_path": "",
        "marker_text_path": "",
        "marker_structure_path": "",
        "fused_text_path": "",
        "fused_structure_path": "",
        "current_text_sha256": "",
        "current_structure_sha256": "",
        "marker_text_sha256": "",
        "marker_structure_sha256": "",
        "fused_text_sha256": "",
        "fused_structure_sha256": "",
        "current_text_length": "0",
        "marker_text_length": "0",
        "fused_text_length": "0",
        "current_block_count": "0",
        "marker_block_count": "0",
        "fusion_contract_version": SCRIPT_CONTRACT_VERSION,
    }
    cur_rel = key2txt.get(key) or (manifest_row.get("text_path") or "")
    if not cur_rel:
        rec.update(status="missing_current_text", note="no key2txt/text_path binding")
        return rec
    cur_text_path = resolve_repo_path(cur_rel)
    if not cur_text_path.exists():
        rec.update(status="missing_current_text", note=f"current text path not found: {cur_text_path}")
        return rec
    cur_text = cur_text_path.read_text(encoding="utf-8", errors="replace")
    rec.update(current_text_path=repo_rel(cur_text_path), current_text_sha256=sha256_file(cur_text_path), current_text_length=str(len(cur_text)))

    cur_struct_rel = (key2structure.get(key) or {}).get("structure_path") or manifest_row.get("structure_path") or ""
    cur_struct: dict[str, Any] = {}
    if cur_struct_rel:
        cur_struct_path = resolve_repo_path(cur_struct_rel)
        if cur_struct_path.exists():
            cur_struct = load_json(cur_struct_path)
            rec.update(current_structure_path=repo_rel(cur_struct_path), current_structure_sha256=sha256_file(cur_struct_path), current_block_count=str(len(cur_struct.get("blocks") or [])))

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
        sp = resolve_repo_path(marker_struct_rel)
        if sp.exists():
            marker_struct = load_json(sp)
            rec.update(marker_structure_path=repo_rel(sp), marker_structure_sha256=sha256_file(sp), marker_block_count=str(len(marker_struct.get("blocks") or [])))

    if not marker_struct and not marker_text:
        rec.update(status="missing_marker", note="current PDF exists but frozen Marker output missing; run run_stage1_marker_pdf_extraction_v1.py")
        return rec

    paper_out = out_root / key
    paper_out.mkdir(parents=True, exist_ok=True)
    fused_text_path = paper_out / "fused_clean_text_v1.md"
    fused_structure_path = paper_out / "fused_structure_v1.json"
    fused_text = build_fused_text(cur_text, marker_text, key)
    fused_payload = {
        "doc_key": key,
        "source_type": "PDF",
        "parser": "current_pdf_plus_marker_pdf",
        "contract_version": SCRIPT_CONTRACT_VERSION,
        "fused_text_path": repo_rel(fused_text_path),
        "fused_text_sha256": sha256_text(fused_text),
        "current_pdf": {
            "text_path": rec["current_text_path"],
            "structure_path": rec["current_structure_path"],
            "text_sha256": rec["current_text_sha256"],
            "structure_sha256": rec["current_structure_sha256"],
            "block_count": int(rec["current_block_count"] or 0),
        },
        "marker_pdf": {
            "text_path": rec["marker_text_path"],
            "structure_path": rec["marker_structure_path"],
            "text_sha256": rec["marker_text_sha256"],
            "structure_sha256": rec["marker_structure_sha256"],
            "block_count": int(rec["marker_block_count"] or 0),
        },
        "blocks": {
            "current_pdf_blocks": cur_struct.get("blocks") or [],
            "marker_pdf_blocks": marker_struct.get("blocks") or [],
        },
        "metadata": {
            "fusion_policy": "preserve current clean text as compatibility surface; attach frozen Marker structure/text additively",
            "no_semantic_inference": True,
            "stage2_stage5_not_run": True,
            "reuse_contract": "rebuild only when current text/structure hashes or marker text/structure hashes change",
        },
    }
    fused_text_path.write_text(fused_text, encoding="utf-8")
    fused_structure_path.write_text(json.dumps(fused_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rec.update(
        status="fused",
        note="OK",
        fused_text_path=repo_rel(fused_text_path),
        fused_structure_path=repo_rel(fused_structure_path),
        fused_text_sha256=sha256_file(fused_text_path),
        fused_structure_sha256=sha256_file(fused_structure_path),
        fused_text_length=str(len(fused_text)),
    )
    return rec


def write_outputs(rows: list[dict[str, str]], summary_out: Path, key2fusion_out: Path) -> None:
    fields = [
        "paper_key", "status", "note", "current_text_path", "current_structure_path",
        "marker_text_path", "marker_structure_path", "fused_text_path", "fused_structure_path",
        "current_text_sha256", "current_structure_sha256", "marker_text_sha256", "marker_structure_sha256",
        "fused_text_sha256", "fused_structure_sha256", "current_text_length", "marker_text_length",
        "fused_text_length", "current_block_count", "marker_block_count", "fusion_contract_version",
    ]
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader(); w.writerows(rows)
    key_fields = ["paper_key", "fused_text_path", "fused_structure_path", "fused_text_sha256", "fused_structure_sha256", "status"]
    # Cumulative governed index: bounded fusion runs update touched keys without
    # deleting prior fused PDF bindings.
    merged: dict[str, dict[str, str]] = {}
    if key2fusion_out.exists():
        with key2fusion_out.open("r", encoding="utf-8", errors="replace", newline="") as f:
            for old in csv.DictReader(f, delimiter="\t"):
                key = (old.get("paper_key") or "").strip()
                if key:
                    merged[key] = {k: old.get(k, "") for k in key_fields}
    for r in rows:
        if r.get("status") == "fused":
            merged[r["paper_key"]] = {k: r.get(k, "") for k in key_fields}
    with key2fusion_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=key_fields, delimiter="\t")
        w.writeheader()
        for key in sorted(merged):
            w.writerow(merged[key])


def main() -> int:
    ap = argparse.ArgumentParser(description="Build governed Stage1 current+Marker PDF fusion artifacts.")
    ap.add_argument("--manifest", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "manifest_current.tsv")
    ap.add_argument("--scope-keys", type=Path, default=None)
    ap.add_argument("--key2txt", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "key2txt.tsv")
    ap.add_argument("--key2structure", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "key2structure.tsv")
    ap.add_argument("--key2marker", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "key2marker_pdf_v1.tsv")
    ap.add_argument("--content-dir", type=Path, default=paths.DATA_CLEANED_CONTENT_DIR)
    ap.add_argument("--summary-out", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "stage1_pdf_current_marker_fusion_summary_v1.tsv")
    ap.add_argument("--key2fusion-out", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "key2stage1_pdf_fusion_v1.tsv")
    args = ap.parse_args()

    scope = read_scope(args.scope_keys)
    manifest = load_manifest_pdf_keys(args.manifest, scope)
    key2txt = load_key2txt(args.key2txt)
    key2structure = load_tsv_by_key(args.key2structure)
    key2marker = load_tsv_by_key(args.key2marker, key_field="paper_key")
    out_root = args.content_dir / "pdf_current_marker_fusion"
    rows: list[dict[str, str]] = []
    for key in sorted(manifest):
        rec = process_one(key, manifest[key], key2txt, key2structure, key2marker, out_root)
        rows.append(rec)
    write_outputs(rows, args.summary_out, args.key2fusion_out)
    fused = sum(1 for r in rows if r["status"] == "fused")
    missing_marker = sum(1 for r in rows if r["status"] == "missing_marker")
    print(f"[OK] fusion summary -> {args.summary_out}")
    print(f"[OK] key2fusion -> {args.key2fusion_out}")
    print(f"[INFO] fused={fused} missing_marker={missing_marker} total={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
