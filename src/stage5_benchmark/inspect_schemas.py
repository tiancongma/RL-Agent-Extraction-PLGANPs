"""
Inspect schemas for weak-label extraction outputs.

Usage examples:
    python src/stage5_benchmark/inspect_schemas.py \
        --tsv "C:\\path\\to\\weak_labels__gemini.tsv" \
        --jsonl "C:\\path\\to\\weak_labels__gemini.jsonl"

    python src/stage5_benchmark/inspect_schemas.py \
        --tsv "data/results/run_xxx/weak_labels__gemini.tsv" \
        --jsonl "data/results/run_xxx/weak_labels__gemini.jsonl" \
        --report "data/benchmark/goren_2025/schema_report__weak_labels.txt" \
        --max-lines 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DOI_REFERENCE_TERMS = ("doi", "reference", "url")
FORMULATION_ID_TERMS = ("formulation", "record_id", "row_id", "sample_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect TSV/JSONL schemas and write a compact schema report."
    )
    parser.add_argument("--tsv", required=True, help="Absolute or relative path to TSV file.")
    parser.add_argument("--jsonl", required=True, help="Absolute or relative path to JSONL file.")
    parser.add_argument(
        "--report",
        default="data/benchmark/goren_2025/schema_report__weak_labels.txt",
        help="Output report path.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=50,
        help="Number of JSONL lines to sample (default: 50).",
    )
    return parser.parse_args()


def find_candidates(names: set[str], terms: tuple[str, ...]) -> list[str]:
    matched: list[str] = []
    for name in sorted(names):
        lower_name = name.lower()
        if any(term in lower_name for term in terms):
            matched.append(name)
    return matched


def collect_key_paths(obj: Any, prefix: str = "", depth: int = 1, max_depth: int = 3) -> set[str]:
    paths: set[str] = set()
    if depth > max_depth:
        return paths

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            current = f"{prefix}.{key_str}" if prefix else key_str
            paths.add(current)
            if isinstance(value, (dict, list)) and depth < max_depth:
                paths.update(collect_key_paths(value, current, depth + 1, max_depth))

    elif isinstance(obj, list):
        for item in obj[:3]:
            if isinstance(item, (dict, list)):
                paths.update(collect_key_paths(item, prefix, depth, max_depth))

    return paths


def inspect_tsv(tsv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(tsv_path, sep="\t", nrows=1)
    columns = [str(c) for c in df.columns.tolist()]
    column_set = set(columns)

    return {
        "columns": columns,
        "doi_reference_candidates": find_candidates(column_set, DOI_REFERENCE_TERMS),
        "formulation_id_candidates": find_candidates(column_set, FORMULATION_ID_TERMS),
    }


def inspect_jsonl(jsonl_path: Path, max_lines: int) -> dict[str, Any]:
    top_level_keys: set[str] = set()
    nested_paths: set[str] = set()
    valid_lines = 0
    parse_errors = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_index, raw_line in enumerate(f, start=1):
            if line_index > max_lines:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            if not isinstance(record, dict):
                continue

            valid_lines += 1
            for key in record.keys():
                top_level_keys.add(str(key))

            nested_paths.update(collect_key_paths(record, max_depth=3))

    return {
        "top_level_keys": sorted(top_level_keys),
        "nested_key_paths": sorted(path for path in nested_paths if "." in path),
        "doi_reference_candidates": find_candidates(top_level_keys.union(nested_paths), DOI_REFERENCE_TERMS),
        "formulation_id_candidates": find_candidates(
            top_level_keys.union(nested_paths), FORMULATION_ID_TERMS
        ),
        "valid_lines": valid_lines,
        "parse_errors": parse_errors,
        "sampled_lines": max_lines,
    }


def build_report(tsv_result: dict[str, Any], jsonl_result: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("Schema Inspection Report: weak_labels__gemini")
    lines.append("")

    lines.append("[TSV] Column names")
    for col in tsv_result["columns"]:
        lines.append(f"- {col}")
    lines.append("")

    lines.append("[TSV] Likely DOI/Reference field candidates")
    if tsv_result["doi_reference_candidates"]:
        for name in tsv_result["doi_reference_candidates"]:
            lines.append(f"- {name}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("[TSV] Likely formulation identifier candidates")
    if tsv_result["formulation_id_candidates"]:
        for name in tsv_result["formulation_id_candidates"]:
            lines.append(f"- {name}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append(
        "[JSONL] Parsed sample summary "
        f"(sampled_lines={jsonl_result['sampled_lines']}, valid_json_objects={jsonl_result['valid_lines']}, "
        f"parse_errors={jsonl_result['parse_errors']})"
    )
    lines.append("")

    lines.append("[JSONL] Top-level keys")
    if jsonl_result["top_level_keys"]:
        for key in jsonl_result["top_level_keys"]:
            lines.append(f"- {key}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("[JSONL] Nested key paths (depth <= 3)")
    if jsonl_result["nested_key_paths"]:
        for path in jsonl_result["nested_key_paths"]:
            lines.append(f"- {path}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("[JSONL] Likely DOI/Reference field candidates")
    if jsonl_result["doi_reference_candidates"]:
        for name in jsonl_result["doi_reference_candidates"]:
            lines.append(f"- {name}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("[JSONL] Likely formulation identifier candidates")
    if jsonl_result["formulation_id_candidates"]:
        for name in jsonl_result["formulation_id_candidates"]:
            lines.append(f"- {name}")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    tsv_path = Path(args.tsv)
    jsonl_path = Path(args.jsonl)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    tsv_result = inspect_tsv(tsv_path)
    jsonl_result = inspect_jsonl(jsonl_path, args.max_lines)

    report = build_report(tsv_result, jsonl_result)
    print(report, end="")

    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
