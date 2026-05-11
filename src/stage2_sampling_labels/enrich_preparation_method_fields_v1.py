#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.utils.preparation_method_fields_v1 import (
    PREPARATION_METHOD_FIELDNAMES,
    enrich_preparation_method_fields_v1,
)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames or []), rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministically append preparation-method enrichment fields to an existing Stage2-style TSV."
    )
    parser.add_argument("--input-tsv", required=True, type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_fieldnames, input_rows = read_rows(args.input_tsv)
    output_fieldnames = [
        *input_fieldnames,
        *[name for name in PREPARATION_METHOD_FIELDNAMES if name not in input_fieldnames],
    ]
    output_rows = [enrich_preparation_method_fields_v1(row) for row in input_rows]
    write_rows(args.output_tsv, output_fieldnames, output_rows)

    populated_preparation_method = sum(
        1 for row in output_rows if str(row.get("preparation_method", "")).strip() not in {"", "unknown"}
    )
    populated_emulsion_structure = sum(
        1 for row in output_rows if str(row.get("emulsion_structure", "")).strip() not in {"", "none"}
    )
    print(
        json.dumps(
            {
                "input_tsv": str(args.input_tsv),
                "output_tsv": str(args.output_tsv),
                "row_count": len(output_rows),
                "populated_preparation_method": populated_preparation_method,
                "populated_emulsion_structure": populated_emulsion_structure,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
