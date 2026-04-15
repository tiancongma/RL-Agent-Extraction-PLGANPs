#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Stage1 cleaner for PDF/HTML with additive structure sidecars.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import trafilatura
except Exception:
    trafilatura = None


ALLOWED_BLOCK_TYPES = {"heading", "paragraph", "table", "caption", "list"}
TABLE_SOURCE_PRIORITY = [
    "existing_html_extraction",
    "marker_extraction",
    "text_detected",
]
INTERNAL_STAGE1_SUMMARY_NAME = "stage1_dual_output_summary.tsv"
TABLE_HEADER_HINTS = [
    r"\b(mean|sd|se|n)\b",
    r"\b(size|diameter|pdi|zeta|ee|dl|loading|encapsulation)\b",
    r"\b(concentration|dose|w/?o|w1|w2|%|ratio)\b",
]
FORBIDDEN_SEMANTIC_KEYS = {
    "formulation_detection",
    "semantic_label",
    "semantic_labels",
    "variable_role",
    "variable_roles",
    "doe_logic",
    "semantic_marker",
    "semantic_markers",
    "inferred_label",
    "inferred_labels",
}


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_inline_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def sha1_text(text: str) -> str:
    return f"sha1:{hashlib.sha1(text.encode('utf-8')).hexdigest()}"


def detect_table_like_text(txt: str) -> bool:
    if not txt:
        return False
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return False
    tabular_like = 0
    num_re = re.compile(r"[-+]?\d+(?:\.\d+)?(?:\s*[x×]\s*\d+(?:\.\d+)?)?")
    for ln in lines[:2000]:
        tokens = re.split(r"[\t|,:; ]{2,}", ln.strip())
        numeric_cells = sum(1 for token in tokens if num_re.search(token))
        if len(tokens) >= 3 and numeric_cells >= 2:
            tabular_like += 1
    header_hit = any(re.search(pattern, txt.lower()) for pattern in TABLE_HEADER_HINTS)
    return tabular_like >= 5 or header_hit


def estimate_parse_quality(txt: str, source_type: str) -> str:
    if not txt:
        return "low"
    lines = [ln for ln in txt.splitlines()]
    if not lines:
        return "low"
    avg_line = sum(len(ln) for ln in lines) / max(1, len(lines))
    hyphens = txt.count("-\n")
    short_lines = sum(1 for ln in lines if 0 < len(ln) < 20)
    ratio_short = short_lines / max(1, len(lines))
    score = 0.0
    score += 1.0 if source_type.upper() == "HTML" else 0.2
    if avg_line > 40:
        score += 0.6
    if ratio_short < 0.3:
        score += 0.4
    if hyphens < 5:
        score += 0.3
    if score >= 1.8:
        return "high"
    if score >= 1.0:
        return "medium"
    return "low"


def to_repo_rel(path_value: Path) -> str:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    try:
        return str(path_value.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except Exception:
        return str(path_value.resolve()).replace("\\", "/")


def ensure_out_dirs(base: Path) -> Tuple[Path, Path, Path]:
    text_dir = base / "text"
    structure_dir = base / "structure"
    text_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    return base, text_dir, structure_dir


def load_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".tsv", ".tab"]:
        return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_MINIMAL, keep_default_na=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def make_block(block_type: str, text: str, page: Optional[int] = None, bbox: Optional[List[float]] = None) -> Dict[str, Any]:
    if block_type not in ALLOWED_BLOCK_TYPES:
        raise ValueError(f"Unsupported block type: {block_type}")
    block: Dict[str, Any] = {
        "type": block_type,
        "page": page,
        "text": text,
    }
    if bbox is not None:
        block["bbox"] = bbox
    return block


def finalize_blocks(raw_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    finalized: List[Dict[str, Any]] = []
    for block in raw_blocks:
        block_type = str(block.get("type", "paragraph"))
        if block_type not in ALLOWED_BLOCK_TYPES:
            continue
        text = str(block.get("text", "") or "")
        text = normalize_whitespace(text) if block_type == "table" else normalize_inline_whitespace(text)
        if not text:
            continue
        clean_block: Dict[str, Any] = {
            "block_id": f"b{len(finalized) + 1:04d}",
            "type": block_type,
            "page": block.get("page"),
            "order": len(finalized) + 1,
            "text": text,
        }
        if block.get("bbox") is not None:
            clean_block["bbox"] = block.get("bbox")
        if block_type == "table" and block.get("table_id"):
            clean_block["table_id"] = str(block["table_id"])
        finalized.append(clean_block)
    return finalized


def project_text_from_blocks(blocks: List[Dict[str, Any]]) -> str:
    chunks = [str(block.get("text", "")).strip() for block in blocks if str(block.get("text", "")).strip()]
    return normalize_whitespace("\n\n".join(chunks))


def classify_bs4_text_block(tag_name: str, text: str) -> str:
    low = tag_name.lower()
    if low in {"caption", "figcaption"}:
        return "caption"
    if low == "li":
        return "list"
    if low in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return "heading"
    return "paragraph"


def classify_xml_text_block(tag_name: str, text: str) -> str:
    low = tag_name.lower()
    if low == "head":
        return "heading"
    if low == "item":
        return "list"
    if low in {"figdesc", "caption"}:
        return "caption"
    return "paragraph"


def serialize_table_rows(rows: List[List[str]]) -> str:
    lines = ["\t".join(normalize_inline_whitespace(cell) for cell in row if normalize_inline_whitespace(cell)) for row in rows]
    lines = [line for line in lines if line.strip()]
    return normalize_whitespace("\n".join(lines))


def load_existing_table_entries(table_dir_value: str, doc_key: str) -> List[Dict[str, Any]]:
    candidate_dirs: List[Path] = []
    if table_dir_value:
        candidate_dirs.append(Path(table_dir_value))

    data_cleaned_root = Path(__file__).resolve().parents[2] / "data" / "cleaned"
    candidate_dirs.extend(
        [
            data_cleaned_root / "content_goren_2025" / "tables" / doc_key,
            data_cleaned_root / "goren_2025" / "tables" / doc_key,
        ]
    )
    for child in data_cleaned_root.iterdir():
        if child.is_dir():
            candidate_dirs.append(child / "tables" / doc_key)

    seen: set[str] = set()
    deduped_dirs: List[Path] = []
    for candidate in candidate_dirs:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped_dirs.append(candidate)

    for table_dir in deduped_dirs:
        if not table_dir.exists() or not table_dir.is_dir():
            continue
        manifest_path = table_dir / "tables_manifest.json"
        selected_files: List[str] = []
        if manifest_path.exists():
            try:
                manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
                selected_files = [str(name) for name in manifest_obj.get("selected_table_files", []) if str(name).strip()]
            except Exception:
                selected_files = []
        if not selected_files:
            selected_files = sorted([p.name for p in table_dir.glob("*.csv")])
        entries: List[Dict[str, Any]] = []
        for idx, file_name in enumerate(selected_files, start=1):
            csv_path = table_dir / file_name
            if not csv_path.exists() or not csv_path.is_file():
                continue
            try:
                df = pd.read_csv(csv_path, dtype=str).fillna("")
                rows = [list(df.columns.astype(str))] + df.astype(str).values.tolist()
            except Exception:
                rows = []
            table_text = serialize_table_rows(rows) if rows else normalize_inline_whitespace(csv_path.name)
            entries.append(
                {
                    "table_id": f"t{idx:03d}",
                    "source_file": to_repo_rel(csv_path),
                    "format": csv_path.suffix.lstrip(".").lower() or "csv",
                    "block_text": table_text or normalize_inline_whitespace(csv_path.name),
                    "source_priority": "existing_html_extraction",
                }
            )
        if entries:
            return entries
    return []


def build_text_detected_table_entries(source_path: Path, txt_path_rel: str, table_texts: List[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    source_file = to_repo_rel(source_path) if source_path.exists() else txt_path_rel
    fmt = source_path.suffix.lstrip(".").lower() if source_path.suffix else "txt"
    for idx, table_text in enumerate(table_texts, start=1):
        clean_text = normalize_whitespace(table_text)
        if not clean_text:
            continue
        entries.append(
            {
                "table_id": f"t{idx:03d}",
                "source_file": source_file,
                "format": fmt or "txt",
                "block_text": clean_text,
                "source_priority": "text_detected",
            }
        )
    return entries


def select_table_entries(
    doc_key: str,
    table_dir_value: str,
    source_path: Path,
    txt_path_rel: str,
    marker_tables: List[str],
    text_detected_tables: List[str],
) -> List[Dict[str, Any]]:
    # One source is selected per table_id by fixed priority. Sources are never merged.
    existing_entries = load_existing_table_entries(table_dir_value=table_dir_value, doc_key=doc_key)
    if existing_entries:
        return existing_entries
    marker_entries = build_text_detected_table_entries(source_path=source_path, txt_path_rel=txt_path_rel, table_texts=marker_tables)
    for entry in marker_entries:
        entry["source_priority"] = "marker_extraction"
    if marker_entries:
        return marker_entries
    return build_text_detected_table_entries(source_path=source_path, txt_path_rel=txt_path_rel, table_texts=text_detected_tables)


def validate_sidecar_payload(payload: Dict[str, Any]) -> None:
    for key in FORBIDDEN_SEMANTIC_KEYS:
        if key in payload:
            raise ValueError(f"Forbidden semantic field emitted: {key}")
    block_types = [block.get("type") for block in payload.get("blocks", [])]
    if any(block_type not in ALLOWED_BLOCK_TYPES for block_type in block_types):
        raise ValueError("Unsupported block type emitted in sidecar")
    table_ids = [str(table["table_id"]) for table in payload.get("tables", [])]
    if len(table_ids) != len(set(table_ids)):
        raise ValueError("Duplicate table_id emitted in tables[]")
    block_table_ids = [str(block["table_id"]) for block in payload.get("blocks", []) if block.get("type") == "table"]
    if any(table_id not in table_ids for table_id in block_table_ids):
        raise ValueError("Table block emitted without matching tables[] entry")
    metadata = payload.get("metadata", {})
    if not {"parser", "parse_quality", "warnings"}.issubset(set(metadata.keys())):
        raise ValueError("metadata missing required fields")


def parse_trafilatura_xml(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    blocks: List[Dict[str, Any]] = []
    for elem in root.iter():
        tag_name = elem.tag.lower()
        if tag_name in {"doc", "main"}:
            continue
        if tag_name in {"p", "head", "item", "caption", "figdesc"}:
            text = normalize_inline_whitespace("".join(elem.itertext()))
            if text:
                blocks.append(make_block(classify_xml_text_block(tag_name, text), text))
    return blocks


def extract_trafilatura_blocks(html_path: Path) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    if trafilatura is None:
        raise RuntimeError("trafilatura_not_available")
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    xml_text = trafilatura.extract(
        html,
        include_tables=True,
        include_formatting=False,
        include_comments=False,
        favor_precision=True,
        output_format="xml",
    )
    if not xml_text:
        raise RuntimeError("trafilatura_empty_output")
    blocks = parse_trafilatura_xml(xml_text)
    if not blocks:
        raise RuntimeError("trafilatura_no_blocks")
    return blocks, "trafilatura_native", []


def extract_bs4_blocks(html_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    blocks: List[Dict[str, Any]] = []
    for elem in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "figcaption", "caption"]):
        text = normalize_inline_whitespace(elem.get_text(separator=" ", strip=True))
        if not text:
            continue
        blocks.append(make_block(classify_bs4_text_block(elem.name, text), text))
    return blocks, []


def extract_text_detected_tables_from_html(html_path: Path) -> List[str]:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    table_texts: List[str] = []
    for table in soup.find_all("table"):
        rows: List[List[str]] = []
        for tr in table.find_all("tr"):
            cells = [normalize_inline_whitespace(cell.get_text(separator=" ", strip=True)) for cell in tr.find_all(["th", "td"])]
            if any(cells):
                rows.append(cells)
        serialized = serialize_table_rows(rows)
        if serialized:
            table_texts.append(serialized)
    return table_texts


def extract_text_from_html(html_path: Path, table_dir_value: str = "") -> Dict[str, Any]:
    warnings: List[str] = []
    try:
        raw_blocks, reading_order_source, parser_warnings = extract_trafilatura_blocks(html_path)
        warnings.extend(parser_warnings)
        parser_name = "trafilatura"
    except Exception as exc:
        raw_blocks, parser_warnings = extract_bs4_blocks(html_path)
        warnings.extend(parser_warnings)
        warnings.append(f"trafilatura_failed:{type(exc).__name__}:{exc}")
        parser_name = "beautifulsoup_fallback"
        reading_order_source = "fallback_linear"

    txt_path_rel = f"text/{html_path.stem}.html.txt"
    text_detected_tables = extract_text_detected_tables_from_html(html_path)
    table_entries = select_table_entries(
        doc_key=html_path.stem,
        table_dir_value=table_dir_value,
        source_path=html_path,
        txt_path_rel=txt_path_rel,
        marker_tables=[],
        text_detected_tables=text_detected_tables,
    )
    if table_entries and parser_name == "trafilatura":
        reading_order_source = "fallback_linear"

    for table_entry in table_entries:
        raw_blocks.append(make_block("table", table_entry["block_text"], page=None))
        raw_blocks[-1]["table_id"] = table_entry["table_id"]

    blocks = finalize_blocks(raw_blocks)
    text = project_text_from_blocks(blocks)
    table_detected = any(block.get("type") == "table" for block in blocks) or detect_table_like_text(text)
    payload = {
        "blocks": blocks,
        "tables": [
            {
                "table_id": entry["table_id"],
                "source_file": entry["source_file"],
                "format": entry["format"],
            }
            for entry in table_entries
        ],
        "metadata": {
            "parser": parser_name,
            "page_count": None,
            "parse_quality": estimate_parse_quality(text, "HTML"),
            "warnings": warnings,
        },
        "reading_order_source": reading_order_source,
    }
    return {
        "text": text,
        "table_detected": table_detected,
        "sidecar": payload,
    }


def extract_marker_pdf_blocks(pdf_path: Path) -> Tuple[List[Dict[str, Any]], int, List[str], List[str]]:
    # Marker is preferred when available. If the import/API is unavailable, the caller falls back.
    try:
        from marker.converters.pdf import PdfConverter  # type: ignore
        from marker.models import create_model_dict  # type: ignore
        from marker.output import text_from_rendered  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"marker_not_available:{type(exc).__name__}:{exc}") from exc

    try:
        artifact_dict = create_model_dict()
    except Exception as exc:
        raise RuntimeError(f"marker_model_init_failed:{type(exc).__name__}:{exc}") from exc

    try:
        converter = PdfConverter(artifact_dict=artifact_dict)
        rendered = converter(str(pdf_path))
    except TypeError as exc:
        raise RuntimeError(f"marker_api_mismatch:{type(exc).__name__}:{exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"marker_runtime_failed:{type(exc).__name__}:{exc}") from exc

    rendered_text = ""
    try:
        rendered_text, _, _ = text_from_rendered(rendered)
    except Exception:
        rendered_text = str(getattr(rendered, "text", "") or getattr(rendered, "markdown", "") or "")
    if not rendered_text.strip():
        raise RuntimeError("marker_empty_output")
    page_count = int(getattr(converter, "page_count", 0) or getattr(rendered, "page_count", 0) or 0)
    blocks: List[Dict[str, Any]] = []
    for chunk in re.split(r"\n{2,}", normalize_whitespace(rendered_text)):
        clean = normalize_inline_whitespace(chunk)
        if clean:
            blocks.append(make_block("paragraph", clean))
    if not blocks:
        raise RuntimeError("marker_no_blocks")
    return blocks, page_count, [], []


def extract_pymupdf_blocks(pdf_path: Path, max_pages: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed. Please `pip install pymupdf`.")
    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
        pages = range(page_count) if max_pages <= 0 else range(min(max_pages, page_count))
        blocks: List[Dict[str, Any]] = []
        for idx in pages:
            page = doc[idx]
            text = page.get_text("text") or ""
            paragraphs = [normalize_inline_whitespace(chunk) for chunk in re.split(r"\n{2,}", text) if normalize_inline_whitespace(chunk)]
            for paragraph in paragraphs:
                blocks.append(make_block("paragraph", paragraph, page=idx + 1))
        return blocks, page_count


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 0, table_dir_value: str = "") -> Dict[str, Any]:
    warnings: List[str] = []
    marker_tables: List[str] = []
    try:
        raw_blocks, page_count, parser_warnings, marker_tables = extract_marker_pdf_blocks(pdf_path)
        warnings.extend(parser_warnings)
        parser_name = "marker"
        reading_order_source = "marker_native"
    except Exception as exc:
        raw_blocks, page_count = extract_pymupdf_blocks(pdf_path, max_pages=max_pages)
        warnings.append(f"marker_failed:{type(exc).__name__}:{exc}")
        parser_name = "pymupdf_fallback"
        reading_order_source = "fallback_linear"

    txt_path_rel = f"text/{pdf_path.stem}.pdf.txt"
    text_detected_tables: List[str] = []
    projected_text = project_text_from_blocks(finalize_blocks(raw_blocks))
    if detect_table_like_text(projected_text):
        text_detected_tables = [projected_text]
    table_entries = select_table_entries(
        doc_key=pdf_path.stem,
        table_dir_value=table_dir_value,
        source_path=pdf_path,
        txt_path_rel=txt_path_rel,
        marker_tables=marker_tables,
        text_detected_tables=text_detected_tables,
    )
    if table_entries and parser_name != "marker":
        reading_order_source = "fallback_linear"

    for table_entry in table_entries:
        raw_blocks.append(make_block("table", table_entry["block_text"], page=None))
        raw_blocks[-1]["table_id"] = table_entry["table_id"]

    blocks = finalize_blocks(raw_blocks)
    text = project_text_from_blocks(blocks)
    table_detected = any(block.get("type") == "table" for block in blocks) or detect_table_like_text(text)
    payload = {
        "blocks": blocks,
        "tables": [
            {
                "table_id": entry["table_id"],
                "source_file": entry["source_file"],
                "format": entry["format"],
            }
            for entry in table_entries
        ],
        "metadata": {
            "parser": parser_name,
            "page_count": page_count,
            "parse_quality": estimate_parse_quality(text, "PDF"),
            "warnings": warnings,
        },
        "reading_order_source": reading_order_source,
    }
    return {
        "text": text,
        "table_detected": table_detected,
        "sidecar": payload,
    }


def build_sidecar_payload(
    key: str,
    source_type: str,
    txt_path: Path,
    text: str,
    parse_payload: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        "doc_key": key,
        "source_type": source_type,
        "txt_path": to_repo_rel(txt_path),
        "txt_hash": sha1_text(text),
        "reading_order_source": parse_payload["reading_order_source"],
        "table_source_priority": list(TABLE_SOURCE_PRIORITY),
        "blocks": parse_payload["blocks"],
        "tables": parse_payload["tables"],
        "metadata": parse_payload["metadata"],
    }
    validate_sidecar_payload(payload)
    return payload


def process_row(
    row: pd.Series,
    out_text_dir: Path,
    out_structure_dir: Path,
    prefer: str,
    single_output: bool,
    max_pages: int,
    overwrite: bool,
    verbose: bool,
) -> List[Dict[str, str]]:
    meta_records: List[Dict[str, str]] = []

    key = (row.get("key") or row.get("id") or "").strip()
    title = (row.get("title") or row.get("name") or "").strip()
    url = (row.get("url") or row.get("link") or "").strip()
    table_dir_value = (row.get("table_dir") or "").strip()

    if not key:
        for candidate in ["key", "id", "uid", "doi", "wosid"]:
            if candidate in row and str(row[candidate]).strip():
                key = str(row[candidate]).strip()
                break

    pdf_col = None
    html_col = None
    for candidate in ["pdf", "pdf_path", "pdffile", "file_pdf"]:
        if candidate in row and str(row[candidate]).strip():
            pdf_col = str(row[candidate]).strip()
            break
    for candidate in ["html", "html_path", "htmlfile", "file_html"]:
        if candidate in row and str(row[candidate]).strip():
            html_col = str(row[candidate]).strip()
            break

    pdf_path = Path(pdf_col) if pdf_col else None
    html_path = Path(html_col) if html_col else None

    if not key:
        return [{
            "key": "",
            "title": title,
            "source_type": "",
            "txt_path": "",
            "structure_path": "",
            "text_length": "0",
            "table_detected": "0",
            "parse_quality": "low",
            "notes": "SKIP: missing key",
            "page_count": "",
            "url": url,
            "parser": "",
            "tables_dir": "",
        }]

    have_pdf = bool(pdf_path and pdf_path.exists() and pdf_path.is_file())
    have_html = bool(html_path and html_path.exists() and html_path.is_file())

    if not have_pdf and not have_html:
        if verbose:
            print(f"[SKIP] No HTML/PDF found for {key}  title={title[:80]}")
        return [{
            "key": key,
            "title": title,
            "source_type": "",
            "txt_path": "",
            "structure_path": "",
            "text_length": "0",
            "table_detected": "0",
            "parse_quality": "low",
            "notes": "SKIP: no input file",
            "page_count": "",
            "url": url,
            "parser": "",
            "tables_dir": table_dir_value,
        }]

    def write_outputs(text: str, source_type: str, sidecar_payload: Dict[str, Any], table_flag: bool) -> Dict[str, str]:
        safe_src = source_type.lower()
        txt_name = f"{key}.{safe_src}.txt"
        json_name = f"{key}.{safe_src}.json"
        txt_path = out_text_dir / txt_name
        structure_path = out_structure_dir / json_name

        if txt_path.exists() and not overwrite:
            prev = txt_path.read_text(encoding="utf-8", errors="ignore")
            page_count = sidecar_payload["metadata"].get("page_count")
            return {
                "key": key,
                "title": title,
                "source_type": source_type,
                "txt_path": str(txt_path.relative_to(out_text_dir.parent)),
                "structure_path": str(structure_path.relative_to(out_structure_dir.parent)) if structure_path.exists() else "",
                "text_length": str(len(prev)),
                "table_detected": "1" if table_flag else "0",
                "parse_quality": estimate_parse_quality(prev, source_type),
                "notes": "OK (skipped write; exists)",
                "page_count": str(page_count) if page_count is not None else "",
                "url": url,
                "parser": str(sidecar_payload["metadata"].get("parser", "")),
                "tables_dir": table_dir_value,
            }

        txt_path.write_text(text, encoding="utf-8")
        sidecar = build_sidecar_payload(
            key=key,
            source_type=source_type,
            txt_path=txt_path,
            text=text,
            parse_payload=sidecar_payload,
        )
        structure_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
        page_count = sidecar["metadata"].get("page_count")
        return {
            "key": key,
            "title": title,
            "source_type": source_type,
            "txt_path": str(txt_path.relative_to(out_text_dir.parent)),
            "structure_path": str(structure_path.relative_to(out_structure_dir.parent)),
            "text_length": str(len(text)),
            "table_detected": "1" if table_flag else "0",
            "parse_quality": str(sidecar["metadata"]["parse_quality"]),
            "notes": "OK",
            "page_count": str(page_count) if page_count is not None else "",
            "url": url,
            "parser": str(sidecar["metadata"]["parser"]),
            "tables_dir": table_dir_value,
        }

    order: List[Tuple[str, Path]] = []
    if have_pdf and have_html:
        order = [("HTML", html_path), ("PDF", pdf_path)] if prefer.lower() == "html" else [("PDF", pdf_path), ("HTML", html_path)]
    elif have_pdf:
        order = [("PDF", pdf_path)]
    else:
        order = [("HTML", html_path)]

    for stype, path_obj in order:
        try:
            if stype == "HTML":
                result = extract_text_from_html(path_obj, table_dir_value=table_dir_value)
                rec = write_outputs(result["text"], "HTML", result["sidecar"], result["table_detected"])
            else:
                result = extract_text_from_pdf(path_obj, max_pages=max_pages, table_dir_value=table_dir_value)
                if not result["text"].strip():
                    raise RuntimeError("no extractable text (scanned/encrypted?)")
                rec = write_outputs(result["text"], "PDF", result["sidecar"], result["table_detected"])
            meta_records.append(rec)
        except Exception as exc:
            meta_records.append(
                {
                    "key": key,
                    "title": title,
                    "source_type": stype,
                    "txt_path": "",
                    "structure_path": "",
                    "text_length": "0",
                    "table_detected": "0",
                    "parse_quality": "low",
                    "notes": f"ERROR: {type(exc).__name__}: {exc}",
                    "page_count": "",
                    "url": url,
                    "parser": "",
                    "tables_dir": table_dir_value,
                }
            )
        if single_output:
            break

    return meta_records


def process_pdf(
    pdf_path: Path,
    outdir: Path,
    keep_sections: List[str] | None = None,
    tables: str = "none",
    debug_trace: bool = False,
    debug_skim: bool = False,
    preview_lines: int = 6,
) -> Tuple[bool, dict]:
    meta = {
        "file": str(pdf_path),
        "out_txt": "",
        "out_structure": "",
        "text_length": 0,
        "page_count": 0,
        "table_detected": 0,
        "parse_quality": "low",
        "notes": "",
    }
    try:
        outdir = Path(outdir)
        _, text_dir, structure_dir = ensure_out_dirs(outdir)
        if not pdf_path.exists():
            meta["notes"] = "ERROR: file not found"
            return False, meta
        result = extract_text_from_pdf(pdf_path, max_pages=0, table_dir_value="")
        text = result["text"]
        if not text.strip():
            meta["notes"] = "ERROR: no extractable text (scanned/encrypted?)"
            return False, meta
        out_txt = text_dir / f"{pdf_path.stem}.pdf.txt"
        out_structure = structure_dir / f"{pdf_path.stem}.pdf.json"
        out_txt.write_text(text, encoding="utf-8")
        payload = build_sidecar_payload(
            key=pdf_path.stem,
            source_type="PDF",
            txt_path=out_txt,
            text=text,
            parse_payload=result["sidecar"],
        )
        out_structure.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        meta.update(
            {
                "out_txt": str(out_txt),
                "out_structure": str(out_structure),
                "text_length": len(text),
                "page_count": int(payload["metadata"].get("page_count") or 0),
                "table_detected": 1 if result["table_detected"] else 0,
                "parse_quality": str(payload["metadata"]["parse_quality"]),
                "notes": "OK",
            }
        )
        return True, meta
    except Exception as exc:
        meta["notes"] = f"ERROR: {type(exc).__name__}: {exc}"
        return False, meta


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Clean text from PDF/HTML and produce a Stage1 internal summary with quality metadata.")
    ap.add_argument("--manifest", required=False, help="Path to CSV/TSV manifest.")
    ap.add_argument("--out-dir", required=False, help="Output base directory (e.g., ./data/cleaned).")
    ap.add_argument("--prefer", choices=["pdf", "html"], default="html", help="If both sources exist, which to prefer as the primary (default: html).")
    ap.add_argument("--single-output", action="store_true", help="Only export the preferred source when both exist.")
    ap.add_argument("--max-pages", type=int, default=0, help="PDF: limit number of pages to extract (0=all).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing text files.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    ap.add_argument("pdf", nargs="?", default=None, help="LEGACY: Single PDF file to clean (if provided, manifest is ignored).")
    ap.add_argument("--outdir", required=False, help="LEGACY alias of --out-dir")
    ap.add_argument("--tables", choices=["camelot", "tabula", "none"], default="none", help="LEGACY: table extraction hint (ignored in manifest mode).")
    ap.add_argument("--debug-trace", action="store_true", help="LEGACY: dump intermediates (ignored in manifest mode).")
    return ap


def main() -> None:
    if len(sys.argv) == 1:
        sys.argv += [
            "--manifest", r".\data\cleaned\manifest.tsv",
            "--out-dir", r".\data\cleaned",
            "--prefer", "html",
            "--overwrite",
            "--verbose",
        ]

    ap = build_arg_parser()
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"[WARN] Ignoring unrecognized arguments: {' '.join(unknown)}")

    out_dir_arg = args.out_dir or args.outdir or r".\data\cleaned"

    if args.pdf:
        pdf_path = Path(args.pdf)
        out_base = Path(out_dir_arg).expanduser().resolve()
        out_base.mkdir(parents=True, exist_ok=True)
        ensure_out_dirs(out_base)
        ok, meta = process_pdf(
            pdf_path=pdf_path,
            outdir=out_base,
            keep_sections=None,
            tables=args.tables,
            debug_trace=args.debug_trace,
            debug_skim=False,
            preview_lines=6,
        )
        status = "[OK]" if ok else "[ERR]"
        print(f"{status} Legacy single-file: {pdf_path.name} -> {meta.get('out_txt', '(no out)')} | notes={meta.get('notes')}")
        return

    if not args.manifest:
        args.manifest = r".\data\cleaned\manifest.tsv"

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_base = Path(out_dir_arg).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    _, out_text_dir, out_structure_dir = ensure_out_dirs(out_base)
    legacy_internal_key2txt = out_base / "key2txt.tsv"
    if legacy_internal_key2txt.exists():
        legacy_internal_key2txt.unlink()

    df = load_manifest(manifest_path)
    if args.verbose:
        print(f"[INFO] Loaded manifest: {manifest_path}  rows={len(df)}")

    all_meta: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        metas = process_row(
            row,
            out_text_dir=out_text_dir,
            out_structure_dir=out_structure_dir,
            prefer=args.prefer,
            single_output=args.single_output,
            max_pages=args.max_pages,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        all_meta.extend(metas)

    out_tsv = out_base / INTERNAL_STAGE1_SUMMARY_NAME
    meta_df = pd.DataFrame(
        all_meta,
        columns=[
            "key",
            "title",
            "source_type",
            "txt_path",
            "structure_path",
            "text_length",
            "table_detected",
            "parse_quality",
            "notes",
            "page_count",
            "url",
            "parser",
            "tables_dir",
        ],
    )
    meta_df.to_csv(out_tsv, sep="\t", index=False)

    ok = (meta_df["notes"].astype(str).str.startswith("OK")).sum()
    skipped = (meta_df["notes"].astype(str).str.startswith("SKIP")).sum()
    errors = (meta_df["notes"].astype(str).str.startswith("ERROR")).sum()
    produced = (meta_df["txt_path"].astype(str).str.len() > 0).sum()

    print(f"[OK] stage1 summary -> {out_tsv}")
    print(f"[INFO] Produced text files: {produced}")
    print(f"[INFO] OK records: {ok} | SKIP: {skipped} | ERROR: {errors}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user.")
