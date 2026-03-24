#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

try:
    from src.utils.paths import DATA_MEM_V1_DIR
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import DATA_MEM_V1_DIR


TYPE_CONFIG = {
    "run": {"file": "run.tsv", "id_field": "run_mem_id", "required": ["run_id"]},
    "lineage": {"file": "lin.tsv", "id_field": "lin_id", "required": ["parent_run", "child_run", "relation"]},
    "decision": {"file": "dec.tsv", "id_field": "dec_id", "required": ["title", "decision"]},
    "error": {"file": "err.tsv", "id_field": "err_id", "required": ["err_sig", "symptom"]},
    "prompt": {"file": "prm.tsv", "id_field": "prm_id", "required": ["title", "recipe"]},
}
STAGE_RE = re.compile(r"\b(stage\s*[0-5]|layer\s*[1-3])\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append one governed memory row to mem_v1.")
    parser.add_argument("--mem-dir", type=Path, default=DATA_MEM_V1_DIR, help="Memory directory. Default: data/mem/v1")
    parser.add_argument("--type", required=True, choices=sorted(TYPE_CONFIG), help="Target memory table type.")
    parser.add_argument("--id", default="", help="Optional explicit stable ID. Default: auto-increment.")
    parser.add_argument("--field", action="append", default=[], help="Repeatable key=value field assignment.")
    return parser.parse_args()


def load_schema(mem_dir: Path) -> dict:
    schema_path = mem_dir / "sch.json"
    if not schema_path.exists():
        raise SystemExit(f"Missing schema manifest: {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_rows(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: str(row.get(key, "")) for key in headers})


def parse_fields(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --field, expected key=value: {value}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --field key: {value}")
        parsed[key] = raw.strip()
    return parsed


def infer_stage(fields: dict[str, str]) -> str:
    blob = " ".join(fields.values())
    match = STAGE_RE.search(blob)
    if not match:
        return fields.get("stage", "")
    return match.group(1).lower().replace(" ", "")


def allocate_id(prefix: str, rows: list[dict[str, str]], id_field: str, explicit_id: str) -> str:
    taken = {str(row.get(id_field, "")).strip() for row in rows if str(row.get(id_field, "")).strip()}
    if explicit_id:
        if explicit_id in taken:
            raise SystemExit(f"ID already exists: {explicit_id}")
        return explicit_id
    max_seen = 0
    for value in taken:
        if value.startswith(prefix):
            suffix = value[len(prefix):]
            if suffix.isdigit():
                max_seen = max(max_seen, int(suffix))
    return f"{prefix}{max_seen + 1:03d}"


def idx_row_from(type_name: str, row: dict[str, str]) -> dict[str, str]:
    if type_name == "run":
        return {
            "mem_id": "",
            "mem_type": "run",
            "ref_id": row["run_mem_id"],
            "stage": row.get("stage", ""),
            "run_id": row.get("run_id", ""),
            "title": row.get("run_id", ""),
            "summary": row.get("summary", "") or row.get("purpose", ""),
            "tags": row.get("run_type", ""),
            "source_file": row.get("source_file", ""),
            "source_kind": row.get("source_kind", ""),
            "status": row.get("status", "active"),
        }
    if type_name == "lineage":
        return {
            "mem_id": "",
            "mem_type": "lineage",
            "ref_id": row["lin_id"],
            "stage": "",
            "run_id": row.get("child_run", ""),
            "title": f"{row.get('child_run', '')} <- {row.get('parent_run', '')}",
            "summary": row.get("note", ""),
            "tags": row.get("relation", ""),
            "source_file": row.get("source_file", ""),
            "source_kind": row.get("source_kind", "manual"),
            "status": "active",
        }
    if type_name == "decision":
        return {
            "mem_id": "",
            "mem_type": "decision",
            "ref_id": row["dec_id"],
            "stage": row.get("stage", ""),
            "run_id": row.get("run_id", ""),
            "title": row.get("title", ""),
            "summary": row.get("decision", ""),
            "tags": row.get("tags", ""),
            "source_file": row.get("source_file", ""),
            "source_kind": row.get("source_kind", ""),
            "status": row.get("status", "active"),
        }
    if type_name == "error":
        return {
            "mem_id": "",
            "mem_type": "error",
            "ref_id": row["err_id"],
            "stage": row.get("stage", ""),
            "run_id": row.get("run_id", ""),
            "title": row.get("err_sig", ""),
            "summary": row.get("symptom", ""),
            "tags": row.get("tags", ""),
            "source_file": row.get("source_file", ""),
            "source_kind": row.get("source_kind", ""),
            "status": row.get("status", "active"),
        }
    return {
        "mem_id": "",
        "mem_type": "prompt",
        "ref_id": row["prm_id"],
        "stage": row.get("stage", ""),
        "run_id": row.get("run_id", ""),
        "title": row.get("title", ""),
        "summary": row.get("recipe", ""),
        "tags": row.get("tags", ""),
        "source_file": row.get("source_file", ""),
        "source_kind": row.get("source_kind", ""),
        "status": row.get("status", "active"),
    }


def main() -> int:
    args = parse_args()
    mem_dir = args.mem_dir.resolve()
    schema = load_schema(mem_dir)
    config = TYPE_CONFIG[args.type]
    fields = parse_fields(args.field)
    table_name = config["file"]
    headers = schema["tables"][table_name]["headers"]
    table_rows = load_rows(mem_dir / table_name)
    idx_rows = load_rows(mem_dir / "idx.tsv")
    for required in config["required"]:
        if not fields.get(required, "").strip():
            raise SystemExit(f"Missing required field for {args.type}: {required}")
    row = {header: fields.get(header, "") for header in headers}
    row["stage"] = row.get("stage", "") or infer_stage(fields)
    if "status" in row and not row["status"]:
        row["status"] = "active"
    if "source_kind" in row and not row["source_kind"]:
        row["source_kind"] = "manual"
    if args.type == "run" and not row.get("summary"):
        row["summary"] = row.get("purpose", "")
    id_field = config["id_field"]
    row[id_field] = allocate_id(schema["tables"][table_name]["id_prefix"], table_rows, id_field, args.id.strip())
    natural_key = tuple(str(row.get(field, "")).strip() for field in config["required"])
    for existing in table_rows:
        other = tuple(str(existing.get(field, "")).strip() for field in config["required"])
        if natural_key == other:
            raise SystemExit(f"Refusing duplicate logical row in {table_name}: {natural_key}")
    table_rows.append(row)
    table_rows.sort(key=lambda item: str(item.get(id_field, "")))
    idx_row = idx_row_from(args.type, row)
    idx_row["mem_id"] = allocate_id(schema["tables"]["idx.tsv"]["id_prefix"], idx_rows, "mem_id", "")
    idx_rows.append(idx_row)
    idx_rows.sort(key=lambda item: str(item.get("mem_id", "")))
    write_rows(mem_dir / table_name, headers, table_rows)
    write_rows(mem_dir / "idx.tsv", schema["tables"]["idx.tsv"]["headers"], idx_rows)
    print(f"table={table_name}")
    print(f"id={row[id_field]}")
    print(f"idx_id={idx_row['mem_id']}")
    print("status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
