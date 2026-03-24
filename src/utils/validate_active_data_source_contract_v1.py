#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

try:
    from src.utils.paths import ACTIVE_RUN_POINTER_FILE, PROJECT_ROOT
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import ACTIVE_RUN_POINTER_FILE, PROJECT_ROOT


MAINTAINED_SCRIPT_SURFACE_FILE = PROJECT_ROOT / "docs" / "maintained_script_surface.tsv"

REQUIRED_POINTER_KEYS = {
    "active_run_id",
    "active_run_dir",
    "authoritative_terminal_files",
    "lineage_policy",
    "updated_at",
    "note",
}

REQUIRED_SCRIPT_SNIPPETS = [
    "resolve_run_context(",
    "resolve_artifact_path(",
]

FORBIDDEN_SCRIPT_SNIPPETS = [
    "sorted(run_dir.glob(\"*scope*.tsv\"))",
    "sorted(run_dir.glob('*scope*.tsv'))",
]


def validate_pointer() -> list[str]:
    errors: list[str] = []
    if not ACTIVE_RUN_POINTER_FILE.exists():
        return [f"Missing active pointer: {ACTIVE_RUN_POINTER_FILE}"]
    payload = json.loads(ACTIVE_RUN_POINTER_FILE.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_POINTER_KEYS - set(payload.keys()))
    if missing:
        errors.append(f"ACTIVE_RUN.json missing keys: {missing}")
    files = payload.get("authoritative_terminal_files")
    if not isinstance(files, dict) or not files:
        errors.append("ACTIVE_RUN.json authoritative_terminal_files must be a non-empty object.")
    return errors


def load_maintained_surface_registry() -> tuple[list[dict[str, str]], list[str]]:
    errors: list[str] = []
    if not MAINTAINED_SCRIPT_SURFACE_FILE.exists():
        return [], [f"Missing maintained script surface registry: {MAINTAINED_SCRIPT_SURFACE_FILE}"]
    with MAINTAINED_SCRIPT_SURFACE_FILE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required_columns = {
        "stage_or_workflow",
        "script_path",
        "role",
        "status",
        "must_use_active_data_source_contract",
        "notes",
    }
    header = set(rows[0].keys()) if rows else set()
    missing_columns = sorted(required_columns - header)
    if missing_columns:
        errors.append(
            f"maintained_script_surface.tsv missing required columns: {missing_columns}"
        )
    if not rows:
        errors.append("maintained_script_surface.tsv must contain at least one row.")
    return rows, errors


def validate_scripts(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        rel_path = Path((row.get("script_path") or "").strip())
        status = (row.get("status") or "").strip()
        must_use_contract = (row.get("must_use_active_data_source_contract") or "").strip().lower()
        if not rel_path:
            errors.append(f"Registry row missing script_path: {row}")
            continue
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            errors.append(f"Registry references missing script: {rel_path}")
            continue
        if path.is_dir():
            errors.append(f"Registry script_path is a directory, not a file: {rel_path}")
            continue
        if status != "maintained_entrypoint":
            continue
        text = path.read_text(encoding="utf-8")
        if must_use_contract == "yes":
            for snippet in REQUIRED_SCRIPT_SNIPPETS:
                if snippet not in text:
                    errors.append(f"{rel_path} missing required snippet: {snippet}")
            for snippet in FORBIDDEN_SCRIPT_SNIPPETS:
                if snippet in text:
                    errors.append(
                        f"{rel_path} still contains forbidden implicit discovery snippet: {snippet}"
                    )
    return errors


def main() -> int:
    rows, registry_errors = load_maintained_surface_registry()
    errors = [*validate_pointer(), *registry_errors, *validate_scripts(rows)]
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"ACTIVE_RUN.json={ACTIVE_RUN_POINTER_FILE}")
    print(f"maintained_script_surface={MAINTAINED_SCRIPT_SURFACE_FILE}")
    for row in rows:
        rel_path = row["script_path"]
        status = row["status"]
        print(f"registry_script={rel_path} status={status}")
        if status == "maintained_entrypoint":
            print(f"validated_script={rel_path}")
    print("status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
