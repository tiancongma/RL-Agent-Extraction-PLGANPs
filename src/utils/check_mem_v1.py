#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    from src.utils.paths import DATA_DIR, DATA_MEM_V1_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import DATA_DIR, DATA_MEM_V1_DIR, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate mem_v1 schema, paths, and row references.")
    parser.add_argument("--mem-dir", type=Path, default=DATA_MEM_V1_DIR, help="Memory directory. Default: data/mem/v1")
    return parser.parse_args()


def load_schema(mem_dir: Path) -> dict:
    schema_path = mem_dir / "sch.json"
    if not schema_path.exists():
        raise SystemExit(f"Missing schema manifest: {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def header_of(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle, delimiter="\t"))


def main() -> int:
    args = parse_args()
    mem_dir = args.mem_dir.resolve()
    schema = load_schema(mem_dir)
    errors: list[str] = []
    rows_by_table: dict[str, list[dict[str, str]]] = {}
    try:
        rel_to_data = mem_dir.relative_to(DATA_DIR.resolve()).parts
        if len(rel_to_data[:-1]) > 1:
            errors.append(f"Memory directory exceeds max depth under data/: {mem_dir}")
    except ValueError:
        rel_to_data = ()
    for path in sorted(mem_dir.iterdir()):
        if path.is_file():
            if len(path.name) > 16:
                errors.append(f"Filename exceeds 16 chars: {path.name}")
            if len(str(path)) > 120:
                errors.append(f"Full path exceeds 120 chars: {path}")
    for name, spec in schema["tables"].items():
        path = mem_dir / name
        if not path.exists():
            errors.append(f"Missing table: {path}")
            continue
        expected = spec["headers"]
        actual = header_of(path)
        if actual != expected:
            errors.append(f"Header mismatch for {name}: expected {expected}, got {actual}")
        rows = load_rows(path)
        rows_by_table[name] = rows
        id_field = expected[0]
        ids = [str(row.get(id_field, "")).strip() for row in rows if str(row.get(id_field, "")).strip()]
        if len(ids) != len(set(ids)):
            errors.append(f"Duplicate IDs in {name}")
        bad_ids = [value for value in ids if not value.startswith(spec["id_prefix"])]
        if bad_ids:
            errors.append(f"Bad ID prefix in {name}: {bad_ids[:3]}")
        for row in rows:
            source_file = str(row.get("source_file", "")).strip()
            if source_file:
                source_path = (PROJECT_ROOT / source_file).resolve()
                if not source_path.exists():
                    errors.append(f"Missing source file referenced by {name}: {source_file}")
    run_ids = {row["run_id"] for row in rows_by_table.get("run.tsv", []) if row.get("run_id")}
    for row in rows_by_table.get("lin.tsv", []):
        if row.get("parent_run") and row["parent_run"] not in run_ids:
            errors.append(f"lin.tsv parent_run missing from run.tsv: {row['parent_run']}")
        if row.get("child_run") and row["child_run"] not in run_ids:
            errors.append(f"lin.tsv child_run missing from run.tsv: {row['child_run']}")
    ref_sets = {
        "run": {row["run_mem_id"] for row in rows_by_table.get("run.tsv", []) if row.get("run_mem_id")},
        "lineage": {row["lin_id"] for row in rows_by_table.get("lin.tsv", []) if row.get("lin_id")},
        "decision": {row["dec_id"] for row in rows_by_table.get("dec.tsv", []) if row.get("dec_id")},
        "error": {row["err_id"] for row in rows_by_table.get("err.tsv", []) if row.get("err_id")},
        "prompt": {row["prm_id"] for row in rows_by_table.get("prm.tsv", []) if row.get("prm_id")},
    }
    for row in rows_by_table.get("idx.tsv", []):
        mem_type = row.get("mem_type", "")
        ref_id = row.get("ref_id", "")
        if mem_type not in ref_sets:
            errors.append(f"idx.tsv unsupported mem_type: {mem_type}")
            continue
        if ref_id not in ref_sets[mem_type]:
            errors.append(f"idx.tsv ref_id not found for {mem_type}: {ref_id}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"mem_dir={mem_dir}")
    for name, rows in rows_by_table.items():
        print(f"{name}={len(rows)}")
    print("status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
