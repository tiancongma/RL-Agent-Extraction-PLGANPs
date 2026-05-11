#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_stage1_marker_pdf_extraction_v1.py

Purpose
- Governed Stage1 Marker PDF extraction entrypoint.
- Freeze Marker outputs under data/cleaned/content instead of rerunning Marker in
  downstream diagnostics.
- Keep Marker as an additive structured PDF surface that can later be fused with
  the current pdf2clean PDF text/structure surface.

Contract
- Input scope is explicit: manifest TSV plus optional paper-key list.
- PDF path resolution reuses Stage1 pdf2clean source-path handling.
- Existing frozen outputs are reused when source hash + config hash match.
- No live LLM usage; Marker's local model path only.
- Writes repo-governed content and index/provenance artifacts:
    data/cleaned/content/marker_pdf/<paper_key>/marker_raw_v1.json
    data/cleaned/content/marker_pdf/<paper_key>/marker_meta_v1.json
    data/cleaned/content/marker_pdf/<paper_key>/marker_clean_text_v1.md
    data/cleaned/content/marker_pdf/<paper_key>/marker_structure_v1.json
    data/cleaned/index/key2marker_pdf_v1.tsv
    data/cleaned/index/stage1_marker_pdf_extraction_summary_v1.tsv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "src" / "utils" / "paths.py").exists():
            sys.path.insert(0, str(p))
            return


_bootstrap_import_paths()
from src.utils import paths  # noqa: E402
from src.stage1_cleaning.pdf2clean import resolve_stage1_source_path  # noqa: E402

MARKER_VERSION = "marker-pdf==1.10.2"
SCRIPT_CONTRACT_VERSION = "stage1_marker_pdf_extraction_v1"
DEFAULT_CONFIG = {
    "output_format": "json",
    "disable_tqdm": True,
    "disable_image_extraction": True,
    "use_llm": False,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(paths.PROJECT_ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def read_scope_keys(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    keys: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        if "\t" in sample or "," in sample:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
            reader = csv.DictReader(f, dialect=dialect)
            if reader.fieldnames and any(h in reader.fieldnames for h in ["paper_key", "key", "zotero_key"]):
                for row in reader:
                    key = (row.get("paper_key") or row.get("key") or row.get("zotero_key") or "").strip()
                    if key:
                        keys.add(key)
                return keys
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s in {"paper_key", "key"}:
                continue
            keys.add(s.split("\t")[0].split(",")[0].strip())
    return keys


def iter_manifest_pdf_rows(manifest: Path, scope_keys: set[str] | None) -> Iterable[dict[str, str]]:
    with manifest.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row.get("paper_key") or row.get("key") or row.get("zotero_key") or "").strip()
            if not key:
                continue
            if scope_keys is not None and key not in scope_keys:
                continue
            pdf_value = (row.get("pdf") or row.get("pdf_path") or row.get("pdffile") or row.get("file_pdf") or "").strip()
            if not pdf_value:
                continue
            yield {"paper_key": key, "title": (row.get("title") or "").strip(), "pdf_value": pdf_value}


def flatten_marker_blocks(node: Any, page_index: int | None = None, out: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if out is None:
        out = []
    if not isinstance(node, dict):
        return out
    block_type = str(node.get("block_type") or "")
    if block_type == "Page":
        page_index = len([b for b in out if b.get("block_type") == "Page"])
    html_value = node.get("html") or ""
    text = ""
    if html_value:
        # Lightweight HTML stripping without requiring bs4 in the runtime Python.
        import re
        text = re.sub(r"<[^>]+>", " ", html_value)
        text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    if block_type and block_type != "Page":
        out.append(
            {
                "block_id": node.get("id") or "",
                "block_type": block_type,
                "page_index": page_index,
                "bbox": node.get("bbox") or [],
                "polygon": node.get("polygon") or [],
                "text": text,
                "section_hierarchy": node.get("section_hierarchy") or {},
            }
        )
    for child in node.get("children") or []:
        flatten_marker_blocks(child, page_index=page_index, out=out)
    return out


def marker_text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for b in blocks:
        text = (b.get("text") or "").strip()
        if not text:
            continue
        bt = b.get("block_type") or ""
        if bt in {"SectionHeader", "Title"}:
            lines.append(f"\n## {text}\n")
        elif bt == "Table":
            lines.append(f"\n[MARKER_TABLE]\n{text}\n")
        else:
            lines.append(text)
    return "\n".join(lines).strip() + ("\n" if lines else "")


def marker_counts(blocks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for b in blocks:
        bt = str(b.get("block_type") or "")
        counts[bt] = counts.get(bt, 0) + 1
    return counts


def config_hash(args: argparse.Namespace) -> str:
    payload = dict(DEFAULT_CONFIG)
    payload.update(
        {
            "marker_executable": str(args.marker_executable),
            "page_range": args.page_range or "",
            "marker_version": MARKER_VERSION,
            "script_contract_version": SCRIPT_CONTRACT_VERSION,
        }
    )
    return sha256_text(json.dumps(payload, sort_keys=True))


def existing_is_current(structure_path: Path, source_hash: str, cfg_hash: str) -> bool:
    if not structure_path.exists():
        return False
    try:
        payload = json.loads(structure_path.read_text(encoding="utf-8", errors="replace"))
        meta = payload.get("metadata") or {}
        return meta.get("source_sha256") == source_hash and meta.get("config_sha256") == cfg_hash
    except Exception:
        return False


def find_marker_json(tmp_out: Path) -> tuple[Path | None, Path | None]:
    json_candidates = sorted(tmp_out.rglob("*.json"))
    raw = None
    meta = None
    for p in json_candidates:
        if p.name.endswith("_meta.json"):
            meta = p
        elif raw is None:
            raw = p
    return raw, meta


def run_marker(marker_executable: Path, pdf_path: Path, tmp_out: Path, page_range: str | None, timeout: int) -> tuple[int, str, str, float]:
    cmd = [
        str(marker_executable),
        str(pdf_path),
        "--output_dir",
        str(tmp_out),
        "--output_format",
        "json",
        "--disable_tqdm",
        "--disable_image_extraction",
    ]
    if page_range:
        cmd += ["--page_range", page_range]
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr, time.time() - start


def process_one(row: dict[str, str], args: argparse.Namespace, cfg_hash: str) -> dict[str, str]:
    key = row["paper_key"]
    pdf_path = resolve_stage1_source_path(row["pdf_value"])
    out_root = args.content_dir / "marker_pdf" / key
    raw_out = out_root / "marker_raw_v1.json"
    meta_out = out_root / "marker_meta_v1.json"
    text_out = out_root / "marker_clean_text_v1.md"
    structure_out = out_root / "marker_structure_v1.json"
    out_root.mkdir(parents=True, exist_ok=True)

    rec = {
        "paper_key": key,
        "title": row.get("title", ""),
        "pdf_path": str(pdf_path),
        "pdf_available": "yes" if pdf_path and pdf_path.exists() else "no",
        "status": "",
        "note": "",
        "marker_raw_path": "",
        "marker_meta_path": "",
        "marker_text_path": "",
        "marker_structure_path": "",
        "source_sha256": "",
        "config_sha256": cfg_hash,
        "text_length": "0",
        "block_count": "0",
        "table_count": "0",
        "section_header_count": "0",
        "elapsed_seconds": "0.000",
    }
    if not pdf_path or not pdf_path.exists():
        rec.update({"status": "missing_pdf", "note": "resolved PDF path not found"})
        return rec

    source_hash = sha256_file(pdf_path)
    rec["source_sha256"] = source_hash
    if not args.overwrite and existing_is_current(structure_out, source_hash, cfg_hash):
        payload = json.loads(structure_out.read_text(encoding="utf-8", errors="replace"))
        counts = payload.get("block_type_counts") or {}
        text_len = len(text_out.read_text(encoding="utf-8", errors="replace")) if text_out.exists() else 0
        rec.update(
            {
                "status": "reused_frozen",
                "note": "existing Marker output reused; source/config hash matched",
                "marker_raw_path": repo_rel(raw_out),
                "marker_meta_path": repo_rel(meta_out) if meta_out.exists() else "",
                "marker_text_path": repo_rel(text_out),
                "marker_structure_path": repo_rel(structure_out),
                "text_length": str(text_len),
                "block_count": str(len(payload.get("blocks") or [])),
                "table_count": str(counts.get("Table", 0)),
                "section_header_count": str(counts.get("SectionHeader", 0)),
            }
        )
        return rec

    with tempfile.TemporaryDirectory(prefix=f"marker_{key}_") as td:
        tmp_out = Path(td)
        try:
            code, stdout, stderr, elapsed = run_marker(args.marker_executable, pdf_path, tmp_out, args.page_range, args.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            rec.update({"status": "marker_timeout", "note": f"timeout after {args.timeout_seconds}s", "elapsed_seconds": f"{args.timeout_seconds:.3f}"})
            return rec
        rec["elapsed_seconds"] = f"{elapsed:.3f}"
        if code != 0:
            note = (stderr or stdout or f"marker exited {code}").strip().replace("\n", " | ")[:1000]
            rec.update({"status": "marker_failed", "note": note})
            return rec
        raw_json, meta_json = find_marker_json(tmp_out)
        if raw_json is None:
            rec.update({"status": "marker_failed", "note": "marker completed but no JSON output was found"})
            return rec
        raw_payload = json.loads(raw_json.read_text(encoding="utf-8", errors="replace"))
        blocks: list[dict[str, Any]] = []
        for child in raw_payload.get("children") or []:
            flatten_marker_blocks(child, out=blocks)
        text = marker_text_from_blocks(blocks)
        counts = marker_counts(blocks)
        structure_payload = {
            "doc_key": key,
            "source_type": "PDF",
            "parser": "marker_pdf",
            "contract_version": SCRIPT_CONTRACT_VERSION,
            "txt_path": repo_rel(text_out),
            "raw_marker_json_path": repo_rel(raw_out),
            "raw_marker_meta_path": repo_rel(meta_out) if meta_json else "",
            "text_sha256": sha256_text(text),
            "blocks": blocks,
            "block_type_counts": counts,
            "metadata": {
                "source_pdf_path": str(pdf_path),
                "source_sha256": source_hash,
                "config_sha256": cfg_hash,
                "marker_version": MARKER_VERSION,
                "marker_executable": str(args.marker_executable),
                "page_range": args.page_range or "",
                "script_contract_version": SCRIPT_CONTRACT_VERSION,
                "elapsed_seconds": elapsed,
                "reuse_contract": "freeze output and reuse unless source_sha256 or config_sha256 changes",
            },
        }
        raw_out.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if meta_json:
            shutil.copyfile(meta_json, meta_out)
        else:
            meta_out.write_text("{}\n", encoding="utf-8")
        text_out.write_text(text, encoding="utf-8")
        structure_out.write_text(json.dumps(structure_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        rec.update(
            {
                "status": "generated",
                "note": "OK",
                "marker_raw_path": repo_rel(raw_out),
                "marker_meta_path": repo_rel(meta_out),
                "marker_text_path": repo_rel(text_out),
                "marker_structure_path": repo_rel(structure_out),
                "text_length": str(len(text)),
                "block_count": str(len(blocks)),
                "table_count": str(counts.get("Table", 0)),
                "section_header_count": str(counts.get("SectionHeader", 0)),
            }
        )
        return rec


def write_summary(rows: list[dict[str, str]], summary_out: Path, key2marker_out: Path) -> None:
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "paper_key", "title", "pdf_path", "pdf_available", "status", "note",
        "marker_raw_path", "marker_meta_path", "marker_text_path", "marker_structure_path",
        "source_sha256", "config_sha256", "text_length", "block_count", "table_count",
        "section_header_count", "elapsed_seconds",
    ]
    with summary_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader(); w.writerows(rows)
    fields2 = ["paper_key", "marker_text_path", "marker_structure_path", "marker_raw_path", "source_sha256", "config_sha256", "status"]
    # Global governed index is cumulative across bounded scopes. Preserve existing
    # frozen bindings and update only keys touched by this run so smoke tests or
    # interrupted batches do not erase previously governed Marker outputs.
    merged: dict[str, dict[str, str]] = {}
    if key2marker_out.exists():
        with key2marker_out.open("r", encoding="utf-8", errors="replace", newline="") as f:
            for old in csv.DictReader(f, delimiter="\t"):
                key = (old.get("paper_key") or "").strip()
                if key:
                    merged[key] = {k: old.get(k, "") for k in fields2}
    for r in rows:
        if r.get("marker_structure_path"):
            merged[r["paper_key"]] = {k: r.get(k, "") for k in fields2}
    with key2marker_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields2, delimiter="\t")
        w.writeheader()
        for key in sorted(merged):
            w.writerow(merged[key])


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Governed Stage1 Marker PDF extraction/freeze entrypoint.")
    ap.add_argument("--manifest", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "manifest_current.tsv")
    ap.add_argument("--scope-keys", type=Path, default=None, help="Optional key list/TSV/CSV scope. Defaults to all manifest PDFs.")
    ap.add_argument("--content-dir", type=Path, default=paths.DATA_CLEANED_CONTENT_DIR)
    ap.add_argument("--summary-out", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "stage1_marker_pdf_extraction_summary_v1.tsv")
    ap.add_argument("--key2marker-out", type=Path, default=paths.DATA_CLEANED_INDEX_DIR / "key2marker_pdf_v1.tsv")
    ap.add_argument("--marker-executable", type=Path, default=paths.PROJECT_ROOT / ".venv_marker" / "bin" / "marker_single")
    ap.add_argument("--page-range", default="", help="Optional Marker page range, e.g. 0,1-3. Empty = full PDF.")
    ap.add_argument("--timeout-seconds", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0, help="Optional first-N PDF limit for smoke tests.")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate even if frozen source/config hashes match.")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.marker_executable.exists():
        raise FileNotFoundError(f"Marker executable not found: {args.marker_executable}. Install with: python3.11 -m venv .venv_marker && .venv_marker/bin/python -m pip install marker-pdf==1.10.2")
    if not args.manifest.exists():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")
    scope = read_scope_keys(args.scope_keys)
    rows = list(iter_manifest_pdf_rows(args.manifest, scope))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    cfg_hash = config_hash(args)
    summary: list[dict[str, str]] = []
    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] marker_pdf {row['paper_key']}", flush=True)
        rec = process_one(row, args, cfg_hash)
        print(f"  -> {rec['status']} blocks={rec['block_count']} tables={rec['table_count']} text={rec['text_length']}", flush=True)
        summary.append(rec)
        write_summary(summary, args.summary_out, args.key2marker_out)
    write_summary(summary, args.summary_out, args.key2marker_out)
    ok = sum(1 for r in summary if r["status"] in {"generated", "reused_frozen"})
    print(f"[OK] marker summary -> {args.summary_out}")
    print(f"[OK] key2marker -> {args.key2marker_out}")
    print(f"[INFO] success_or_reused={ok} total={len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
