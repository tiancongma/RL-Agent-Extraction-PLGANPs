#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_RUNBOOK = ROOT / "project" / "ACTIVE_PIPELINE_RUNBOOK.md"
PIPELINE_MAP = ROOT / "project" / "PIPELINE_SCRIPT_MAP.md"
TOOL_INDEX = ROOT / "docs" / "tool_index.md"
SRC_DIR = ROOT / "src"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_active_runbook_entries(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    pattern = re.compile(
        r"- `script_path`: `(?P<script>src/[^`]+\.py)`\s+"
        r"- `status`: `(?P<status>[A-Z_]+)`",
        flags=re.MULTILINE,
    )
    for match in pattern.finditer(text):
        entries.append(
            {
                "script_path": match.group("script"),
                "status": match.group("status"),
            }
        )
    return entries


def parse_tool_index_paths(text: str) -> set[str]:
    return set(re.findall(r"`(src/[^`]+\.py)`", text))


def parse_pipeline_map_script_refs(text: str) -> set[str]:
    refs: set[str] = set(re.findall(r"`(src/[^`]+\.py)`", text))
    table_script_pattern = re.compile(r"^\|\s*([A-Za-z0-9_./-]+\.py)\s*\|", flags=re.MULTILINE)
    for match in table_script_pattern.finditer(text):
        name = match.group(1)
        if name.startswith("src/"):
            refs.add(name)
            continue
        matches = list(SRC_DIR.rglob(name))
        for path in matches:
            refs.add(path.relative_to(ROOT).as_posix())
    return refs


def collect_all_src_python() -> set[str]:
    return {path.relative_to(ROOT).as_posix() for path in SRC_DIR.rglob("*.py")}


def format_paths(paths: Iterable[str]) -> list[str]:
    return [f"  - {path}" for path in sorted(set(paths))]


def marker(ok: bool, *, warn: bool = False) -> str:
    if warn:
        return "[WARN]"
    return "[OK]" if ok else "[ERR]"


def main() -> int:
    active_text = read_text(ACTIVE_RUNBOOK)
    pipeline_text = read_text(PIPELINE_MAP)
    tool_index_text = read_text(TOOL_INDEX)

    active_entries = parse_active_runbook_entries(active_text)
    active_mainline = sorted(
        {
            entry["script_path"]
            for entry in active_entries
            if entry["status"] == "ACTIVE_MAINLINE"
        }
    )
    tool_index_paths = parse_tool_index_paths(tool_index_text)
    pipeline_map_refs = parse_pipeline_map_script_refs(pipeline_text)
    all_src_python = collect_all_src_python()

    missing_active = [path for path in active_mainline if not (ROOT / path).exists()]
    active_missing_from_tool_index = [path for path in active_mainline if path not in tool_index_paths]
    missing_pipeline_map = [path for path in sorted(pipeline_map_refs) if not (ROOT / path).exists()]
    unregistered_src = sorted(all_src_python - tool_index_paths)

    ok_active = not missing_active
    ok_pipeline = not missing_pipeline_map
    ok_tool_index = not active_missing_from_tool_index
    has_warning = bool(unregistered_src)

    print("RUNBOOK CHECK")
    print(f"{marker(ok_active)} active scripts exist")
    print(f"{marker(ok_tool_index)} active scripts are registered in docs/tool_index.md")
    print(f"{marker(ok_pipeline)} pipeline map scripts exist")
    if has_warning:
        print(f"{marker(False, warn=True)} unregistered scripts found")
    else:
        print(f"{marker(True)} no unregistered scripts found")

    if missing_active:
        print("Missing ACTIVE_MAINLINE scripts:")
        print("\n".join(format_paths(missing_active)))
    if active_missing_from_tool_index:
        print("ACTIVE_MAINLINE scripts missing from docs/tool_index.md:")
        print("\n".join(format_paths(active_missing_from_tool_index)))
    if missing_pipeline_map:
        print("PIPELINE_SCRIPT_MAP.md references missing scripts:")
        print("\n".join(format_paths(missing_pipeline_map)))
    if unregistered_src:
        print("Python scripts under src/ not registered in docs/tool_index.md:")
        print("\n".join(format_paths(unregistered_src)))

    inconsistent = bool(missing_active or active_missing_from_tool_index or missing_pipeline_map or unregistered_src)
    return 1 if inconsistent else 0


if __name__ == "__main__":
    raise SystemExit(main())
