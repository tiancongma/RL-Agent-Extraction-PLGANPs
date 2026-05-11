#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic Stage1 parser structure-quality scorer.

Consumes explicit parser-bakeoff artifacts and emits diagnostic-only structure
metrics. This script does not modify active Stage1 cleaned text, table assets, or
runtime parser behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

PAPER_COLUMNS = [
    "paper_key",
    "parser",
    "parser_variant",
    "status",
    "text_chars",
    "word_count",
    "body_block_count",
    "table_count",
    "cell_count",
    "nonempty_cell_count",
    "max_columns",
    "median_columns",
    "header_cell_count",
    "header_path_coverage",
    "blank_cell_count",
    "caption_bound_table_count",
    "page_locator_coverage",
    "bbox_coverage",
    "anchor_exact_hits",
    "anchor_numeric_fallback_hits",
    "anchor_missing_fragments",
    "geometry_warning_count",
    "benchmark_valid",
]

TABLE_COLUMNS = [
    "paper_key",
    "parser",
    "parser_variant",
    "table_id",
    "table_counted",
    "cell_count",
    "nonempty_cell_count",
    "row_count",
    "max_columns",
    "header_cell_count",
    "header_path_coverage",
    "blank_cell_count",
    "caption_bound",
    "multirow_header_detected",
    "rowspan_cell_count",
    "colspan_cell_count",
    "page_locator_coverage",
    "bbox_coverage",
    "geometry_warning_count",
    "benchmark_valid",
]

DELTA_COLUMNS = [
    "paper_key",
    "baseline_parser",
    "candidate_parser",
    "metric",
    "baseline_value",
    "candidate_value",
    "delta",
    "benchmark_valid",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_tsv(path: Path, rows: Iterable[Dict[str, Any]], columns: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: stringify(row.get(k, "")) for k in columns})


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def key_parser(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row.get("paper_key", "")), str(row.get("parser", ""))


def key_parser_table(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return str(row.get("paper_key", "")), str(row.get("parser", "")), str(row.get("table_id", ""))


def is_yes(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1", "y"}


def nonempty_cell(cell: Dict[str, Any]) -> bool:
    return bool(str(cell.get("normalized_cell_text") or cell.get("raw_cell_text") or "").strip())


def parse_json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def has_header_path(cell: Dict[str, Any]) -> bool:
    return bool(parse_json_list(cell.get("header_path_json")))


def expanded_col_end(cell: Dict[str, Any]) -> int:
    try:
        col = int(cell.get("col_index") or 0)
    except Exception:
        col = 0
    try:
        colspan = int(cell.get("colspan") or 1)
    except Exception:
        colspan = 1
    return col + max(1, colspan) - 1


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"[-+]?\d+(?:\.\d+)?(?:%|[a-zA-Z/]+)?", text or ""))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def load_anchor_fragments(path: Optional[Path]) -> Dict[str, List[Dict[str, str]]]:
    if not path:
        return {}
    anchors: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in read_tsv(path):
        paper_key = (row.get("paper_key") or row.get("key") or "").strip()
        fragment = (row.get("fragment") or row.get("anchor_text") or row.get("text") or "").strip()
        if paper_key and fragment:
            anchors[paper_key].append(row)
    return anchors


def score_anchors_for_text(text: str, anchors: List[Dict[str, str]]) -> Tuple[int, int, int]:
    normalized = normalize_text(text)
    exact = 0
    numeric_fallback = 0
    missing = 0
    for row in anchors:
        fragment = row.get("fragment") or row.get("anchor_text") or row.get("text") or ""
        norm_fragment = normalize_text(fragment)
        if norm_fragment and norm_fragment in normalized:
            exact += 1
            continue
        nums = numeric_tokens(fragment)
        if nums and all(num.lower() in normalized for num in nums):
            numeric_fallback += 1
        else:
            missing += 1
    return exact, numeric_fallback, missing


def cell_has_page(cell: Dict[str, Any]) -> bool:
    return str(cell.get("page", "")).strip() not in {"", "None", "null"}


def cell_has_bbox(cell: Dict[str, Any]) -> bool:
    bbox = cell.get("bbox_json") or cell.get("bbox") or ""
    return str(bbox).strip() not in {"", "None", "null", "[]", "{}"}


def ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0"
    return f"{numerator / denominator:.6f}"


def word_count(text: str) -> int:
    return len(re.findall(r"\b\S+\b", text or ""))


def group_text_by_key(block_rows: List[Dict[str, Any]]) -> Tuple[Dict[Tuple[str, str], str], Counter]:
    texts: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    body_counts: Counter = Counter()
    for row in block_rows:
        kp = key_parser(row)
        texts[kp].append(str(row.get("text", "")))
        block_type = str(row.get("type", "")).lower()
        if block_type not in {"table", "caption", "table_caption"}:
            body_counts[kp] += 1
    return {kp: "\n".join(parts) for kp, parts in texts.items()}, body_counts


def score_table(cells: List[Dict[str, Any]], warning_count: int, parser_variant: str) -> Dict[str, Any]:
    first = cells[0] if cells else {}
    rows = {str(c.get("row_index", "")) for c in cells if str(c.get("row_index", "")).strip()}
    header_rows = {str(c.get("row_index", "")) for c in cells if is_yes(c.get("is_header_cell"))}
    header_count = sum(1 for c in cells if is_yes(c.get("is_header_cell")))
    header_path_count = sum(1 for c in cells if has_header_path(c))
    page_count = sum(1 for c in cells if cell_has_page(c))
    bbox_count = sum(1 for c in cells if cell_has_bbox(c))
    return {
        "paper_key": first.get("paper_key", ""),
        "parser": first.get("parser", ""),
        "parser_variant": parser_variant or first.get("parser_variant", ""),
        "table_id": first.get("table_id", ""),
        "table_counted": "yes",
        "cell_count": len(cells),
        "nonempty_cell_count": sum(1 for c in cells if nonempty_cell(c)),
        "row_count": len(rows),
        "max_columns": max([expanded_col_end(c) for c in cells] or [0]),
        "header_cell_count": header_count,
        "header_path_coverage": ratio(header_path_count, len(cells)),
        "blank_cell_count": sum(1 for c in cells if not nonempty_cell(c)),
        "caption_bound": "yes" if any(str(c.get("caption", "")).strip() for c in cells) else "no",
        "multirow_header_detected": "yes" if len(header_rows) >= 2 else "no",
        "rowspan_cell_count": sum(1 for c in cells if int(c.get("rowspan") or 1) > 1),
        "colspan_cell_count": sum(1 for c in cells if int(c.get("colspan") or 1) > 1),
        "page_locator_coverage": ratio(page_count, len(cells)),
        "bbox_coverage": ratio(bbox_count, len(cells)),
        "geometry_warning_count": warning_count,
        "benchmark_valid": "no",
    }


def numeric(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def build_delta_rows(paper_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_paper: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in paper_rows:
        by_paper[str(row.get("paper_key", ""))][str(row.get("parser", ""))] = row
    metrics = [
        "text_chars",
        "word_count",
        "body_block_count",
        "table_count",
        "cell_count",
        "nonempty_cell_count",
        "max_columns",
        "header_cell_count",
        "blank_cell_count",
        "caption_bound_table_count",
        "anchor_exact_hits",
        "anchor_numeric_fallback_hits",
        "anchor_missing_fragments",
        "geometry_warning_count",
    ]
    out: List[Dict[str, Any]] = []
    for paper_key, parser_rows in sorted(by_paper.items()):
        if "current" not in parser_rows:
            continue
        base = parser_rows["current"]
        for parser, candidate in sorted(parser_rows.items()):
            if parser == "current":
                continue
            for metric in metrics:
                base_value = numeric(base.get(metric))
                cand_value = numeric(candidate.get(metric))
                out.append(
                    {
                        "paper_key": paper_key,
                        "baseline_parser": "current",
                        "candidate_parser": parser,
                        "metric": metric,
                        "baseline_value": base.get(metric, ""),
                        "candidate_value": candidate.get(metric, ""),
                        "delta": cand_value - base_value,
                        "benchmark_valid": "no",
                    }
                )
    return out


def run_structure_quality_scoring(bakeoff_dir: Path, out_dir: Path, anchor_fragments_tsv: Optional[Path] = None) -> Dict[str, Any]:
    bakeoff_dir = bakeoff_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_tsv(bakeoff_dir / "parser_bakeoff_summary_v1.tsv")
    block_rows = read_jsonl(bakeoff_dir / "parser_bakeoff_blocks_v1.jsonl")
    cell_rows = read_jsonl(bakeoff_dir / "parser_bakeoff_cells_v1.jsonl")
    warning_rows = read_tsv(bakeoff_dir / "parser_bakeoff_warnings_v1.tsv")
    anchors = load_anchor_fragments(anchor_fragments_tsv.expanduser().resolve() if anchor_fragments_tsv else None)

    text_by_key, body_counts = group_text_by_key(block_rows)
    warnings_by_key: Counter = Counter(key_parser(row) for row in warning_rows)
    cells_by_table: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for cell in cell_rows:
        cells_by_table[key_parser_table(cell)].append(cell)

    parser_variant_by_key: Dict[Tuple[str, str], str] = {}
    status_by_key: Dict[Tuple[str, str], str] = {}
    for row in summary_rows:
        kp = key_parser(row)
        parser_variant_by_key[kp] = row.get("parser_variant", "")
        status_by_key[kp] = row.get("status", "")

    table_metric_rows: List[Dict[str, Any]] = []
    tables_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for kpt, cells in sorted(cells_by_table.items()):
        kp = (kpt[0], kpt[1])
        table_row = score_table(cells, warnings_by_key[kp], parser_variant_by_key.get(kp, ""))
        table_metric_rows.append(table_row)
        tables_by_key[kp].append(table_row)

    all_keys = sorted({key_parser(r) for r in summary_rows} | set(text_by_key) | set(tables_by_key))
    paper_metric_rows: List[Dict[str, Any]] = []
    for kp in all_keys:
        paper_key, parser = kp
        text = text_by_key.get(kp, "")
        tables = tables_by_key.get(kp, [])
        cell_count = sum(int(t.get("cell_count") or 0) for t in tables)
        header_count = sum(int(t.get("header_cell_count") or 0) for t in tables)
        header_path_count = sum(int(round(float(t.get("header_path_coverage") or 0) * int(t.get("cell_count") or 0))) for t in tables)
        page_cell_count = sum(int(round(float(t.get("page_locator_coverage") or 0) * int(t.get("cell_count") or 0))) for t in tables)
        bbox_cell_count = sum(int(round(float(t.get("bbox_coverage") or 0) * int(t.get("cell_count") or 0))) for t in tables)
        max_columns_values = [int(t.get("max_columns") or 0) for t in tables]
        exact, numeric_fallback, missing = score_anchors_for_text(text, anchors.get(paper_key, []))
        paper_metric_rows.append(
            {
                "paper_key": paper_key,
                "parser": parser,
                "parser_variant": parser_variant_by_key.get(kp, ""),
                "status": status_by_key.get(kp, ""),
                "text_chars": len(text),
                "word_count": word_count(text),
                "body_block_count": body_counts[kp],
                "table_count": len(tables),
                "cell_count": cell_count,
                "nonempty_cell_count": sum(int(t.get("nonempty_cell_count") or 0) for t in tables),
                "max_columns": max(max_columns_values or [0]),
                "median_columns": median(max_columns_values) if max_columns_values else 0,
                "header_cell_count": header_count,
                "header_path_coverage": ratio(header_path_count, cell_count),
                "blank_cell_count": sum(int(t.get("blank_cell_count") or 0) for t in tables),
                "caption_bound_table_count": sum(1 for t in tables if t.get("caption_bound") == "yes"),
                "page_locator_coverage": ratio(page_cell_count, cell_count),
                "bbox_coverage": ratio(bbox_cell_count, cell_count),
                "anchor_exact_hits": exact,
                "anchor_numeric_fallback_hits": numeric_fallback,
                "anchor_missing_fragments": missing,
                "geometry_warning_count": warnings_by_key[kp],
                "benchmark_valid": "no",
            }
        )

    delta_rows = build_delta_rows(paper_metric_rows)
    write_tsv(out_dir / "stage1_structure_quality_by_paper_v1.tsv", paper_metric_rows, PAPER_COLUMNS)
    write_tsv(out_dir / "stage1_structure_quality_by_table_v1.tsv", table_metric_rows, TABLE_COLUMNS)
    write_tsv(out_dir / "stage1_structure_quality_delta_v1.tsv", delta_rows, DELTA_COLUMNS)
    write_run_context(out_dir, bakeoff_dir, anchor_fragments_tsv, len(paper_metric_rows), len(table_metric_rows))
    return {"status": "completed", "out_dir": str(out_dir), "paper_rows": len(paper_metric_rows), "table_rows": len(table_metric_rows)}


def write_run_context(out_dir: Path, bakeoff_dir: Path, anchor_fragments_tsv: Optional[Path], paper_rows: int, table_rows: int) -> None:
    content = "\n".join(
        [
            "# Stage1 Structure Quality Diagnostic RUN_CONTEXT",
            "",
            f"generated_at: {utc_now_iso()}",
            "diagnostic_only: yes",
            "benchmark_valid: no",
            "live_llm_calls: no",
            "active_run_json_modified: no",
            "runtime_parser_behavior_modified: no",
            "script: src/stage1_cleaning/score_stage1_structure_quality_v1.py",
            f"input_bakeoff_dir: {bakeoff_dir}",
            f"anchor_fragments_tsv: {anchor_fragments_tsv or ''}",
            f"output_dir: {out_dir}",
            f"paper_metric_rows: {paper_rows}",
            f"table_metric_rows: {table_rows}",
            "",
            "Outputs:",
            "- stage1_structure_quality_by_paper_v1.tsv",
            "- stage1_structure_quality_by_table_v1.tsv",
            "- stage1_structure_quality_delta_v1.tsv",
            "",
        ]
    )
    (out_dir / "RUN_CONTEXT.md").write_text(content, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Score diagnostic Stage1 parser bakeoff structure quality.")
    ap.add_argument("--bakeoff-dir", type=Path, required=True, help="Explicit parser bakeoff directory containing parser_bakeoff_* artifacts.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Explicit diagnostic output directory.")
    ap.add_argument("--anchor-fragments-tsv", type=Path, default=None, help="Optional explicit TSV with paper_key and fragment/anchor_text columns.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_structure_quality_scoring(
        bakeoff_dir=args.bakeoff_dir,
        out_dir=args.out_dir,
        anchor_fragments_tsv=args.anchor_fragments_tsv,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
