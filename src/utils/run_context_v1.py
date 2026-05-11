from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


REQUIRED_RUN_CONTEXT_FIELDS = [
    "run_class",
    "stage_coverage",
    "boundary_class",
    "lawful_resume_boundary",
    "upstream_authority_source",
    "created_by_script",
]

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "run_class": ("run_class", "run_type"),
    "stage_coverage": ("stage_coverage", "current_stage_boundary"),
    "boundary_class": ("boundary_class",),
    "lawful_resume_boundary": ("lawful_resume_boundary",),
    "upstream_authority_source": ("upstream_authority_source",),
    "created_by_script": ("created_by_script", "stage_local_owner_script"),
}

SECTION_TITLE_PATTERN = re.compile(r"^##\s+(?P<title>.+?)\s*$")
KEY_VALUE_BULLET_PATTERN = re.compile(r"^- (?P<key>[^:]+):\s*`(?P<value>.*)`\s*$")
BARE_CODE_LINE_PATTERN = re.compile(r"^`(?P<value>.*)`\s*$")
BARE_CODE_BULLET_PATTERN = re.compile(r"^- `(?P<value>.*)`\s*$")
FIRST_SCRIPT_PATH_PATTERN = re.compile(r"(src/[A-Za-z0-9_./-]+\.py)")


def _normalize_text(value: object) -> str:
    return str(value or "").strip()


def _normalize_key(value: str) -> str:
    normalized = _normalize_text(value).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def _split_sections(lines: list[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    bullet_map: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current_section = ""
    for raw_line in lines:
        line = raw_line.rstrip()
        match = SECTION_TITLE_PATTERN.match(line.strip())
        if match:
            current_section = _normalize_key(match.group("title"))
            sections.setdefault(current_section, [])
            continue
        if current_section:
            sections[current_section].append(line)
        bullet_match = KEY_VALUE_BULLET_PATTERN.match(line.strip())
        if bullet_match:
            bullet_map[_normalize_key(bullet_match.group("key"))] = _normalize_text(
                bullet_match.group("value")
            )
    return bullet_map, sections


def _first_code_value(section_lines: Iterable[str]) -> str:
    for line in section_lines:
        stripped = line.strip()
        bare_code = BARE_CODE_LINE_PATTERN.match(stripped)
        if bare_code:
            return _normalize_text(bare_code.group("value"))
        bare_bullet = BARE_CODE_BULLET_PATTERN.match(stripped)
        if bare_bullet:
            return _normalize_text(bare_bullet.group("value"))
    return ""


def _section_key_value(section_lines: Iterable[str], key: str) -> str:
    wanted = _normalize_key(key)
    for line in section_lines:
        match = KEY_VALUE_BULLET_PATTERN.match(line.strip())
        if not match:
            continue
        if _normalize_key(match.group("key")) == wanted:
            return _normalize_text(match.group("value"))
    return ""


def _section_text(section_lines: Iterable[str]) -> str:
    values: list[str] = []
    for line in section_lines:
        stripped = line.strip()
        if stripped:
            values.append(stripped)
    return "\n".join(values)


def _extract_stage_coverage(
    bullet_map: dict[str, str],
    sections: dict[str, list[str]],
    text: str,
) -> str:
    for alias in FIELD_ALIASES["stage_coverage"]:
        if bullet_map.get(alias):
            return bullet_map[alias]
    for section_name in ("stage_coverage", "3_stage_coverage", "stage_boundary", "4_stage_boundary"):
        if sections.get(section_name):
            explicit = _section_key_value(sections[section_name], "stage_coverage")
            if explicit:
                return explicit
            first_code = _first_code_value(sections[section_name])
            if first_code:
                return first_code
            boundary = _section_key_value(sections[section_name], "current_stage_boundary")
            if boundary:
                return boundary
    benchmark_match = re.search(
        r"executed the .*?(stage2.*?stage5.*?) chain",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if benchmark_match:
        return re.sub(r"\s+", " ", benchmark_match.group(1)).strip()
    return ""


def _extract_boundary_class(
    bullet_map: dict[str, str],
    sections: dict[str, list[str]],
    text: str,
) -> str:
    explicit = bullet_map.get("boundary_class")
    if explicit:
        return explicit
    for section_name in ("boundary_classification", "4_stage_boundary", "stage_boundary"):
        if not sections.get(section_name):
            continue
        section_lines = sections[section_name]
        explicit = _section_key_value(section_lines, "boundary_class")
        if explicit:
            return explicit
        summary_parts: list[str] = []
        for line in section_lines:
            match = KEY_VALUE_BULLET_PATTERN.match(line.strip())
            if match:
                normalized = _normalize_key(match.group("key"))
                if normalized.startswith("stage"):
                    summary_parts.append(
                        f"{normalized}={_normalize_text(match.group('value'))}"
                    )
        if summary_parts:
            return "; ".join(summary_parts)
    if "benchmark-facing" in text.lower():
        return "benchmark_facing_final_output"
    return ""


def _extract_lawful_resume_boundary(
    bullet_map: dict[str, str],
    sections: dict[str, list[str]],
    text: str,
) -> str:
    explicit = bullet_map.get("lawful_resume_boundary")
    if explicit:
        return explicit
    for section_name in ("stage_boundary", "4_stage_boundary", "benchmark_contract"):
        if not sections.get(section_name):
            continue
        explicit = _section_key_value(sections[section_name], "lawful_resume_boundary")
        if explicit:
            return explicit
    lowered = text.lower()
    if "not a lawful stage3 resume boundary" in lowered:
        return "no"
    if "benchmark-facing" in lowered:
        return "stage5_final_output"
    return ""


def _extract_upstream_authority_source(
    bullet_map: dict[str, str],
    sections: dict[str, list[str]],
) -> str:
    explicit = bullet_map.get("upstream_authority_source")
    if explicit:
        return explicit
    for section_name in ("source_run", "source_resolution", "starting_inputs", "4_starting_inputs"):
        if not sections.get(section_name):
            continue
        section_lines = sections[section_name]
        for key in (
            "upstream_authority_source",
            "active_baseline_run_resolved_through",
            "step1_run_dir",
            "source_run",
            "manifest_tsv",
        ):
            value = _section_key_value(section_lines, key)
            if value:
                return value
        first_code = _first_code_value(section_lines)
        if first_code:
            return first_code
    return ""


def _extract_created_by_script(
    bullet_map: dict[str, str],
    sections: dict[str, list[str]],
    text: str,
) -> str:
    for alias in FIELD_ALIASES["created_by_script"]:
        if bullet_map.get(alias):
            return bullet_map[alias]
    for section_name in ("stage_boundary", "4_stage_boundary", "script_paths_used", "6_script_paths_used"):
        if not sections.get(section_name):
            continue
        section_text = _section_text(sections[section_name])
        match = FIRST_SCRIPT_PATH_PATTERN.search(section_text.replace("\\", "/"))
        if match:
            return match.group(1)
    command_match = re.search(
        r"(?:python|py(?:thon)?(?:\.exe)?)\s+(src/[A-Za-z0-9_./-]+\.py)",
        text.replace("\\", "/"),
        flags=re.IGNORECASE,
    )
    if command_match:
        return command_match.group(1)
    return ""


def extract_run_context_fields(path: str | Path) -> dict[str, object]:
    context_path = Path(path).resolve()
    if context_path.is_dir():
        context_path = context_path / "RUN_CONTEXT.md"
    if not context_path.exists():
        raise FileNotFoundError(f"RUN_CONTEXT.md not found: {context_path}")
    if not context_path.is_file():
        raise FileNotFoundError(f"RUN_CONTEXT.md is not a file: {context_path}")

    text = context_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    bullet_map, sections = _split_sections(lines)
    normalized = {
        "run_class": "",
        "stage_coverage": "",
        "boundary_class": "",
        "lawful_resume_boundary": "",
        "upstream_authority_source": "",
        "created_by_script": "",
    }

    for alias in FIELD_ALIASES["run_class"]:
        if bullet_map.get(alias):
            normalized["run_class"] = bullet_map[alias]
            break
    if not normalized["run_class"]:
        for section_name in ("run_type", "2_run_type", "2_run_type_run_type"):
            if sections.get(section_name):
                normalized["run_class"] = _first_code_value(sections[section_name])
                if normalized["run_class"]:
                    break

    normalized["stage_coverage"] = _extract_stage_coverage(bullet_map, sections, text)
    normalized["boundary_class"] = _extract_boundary_class(bullet_map, sections, text)
    normalized["lawful_resume_boundary"] = _extract_lawful_resume_boundary(
        bullet_map, sections, text
    )
    normalized["upstream_authority_source"] = _extract_upstream_authority_source(
        bullet_map, sections
    )
    normalized["created_by_script"] = _extract_created_by_script(bullet_map, sections, text)

    missing = [field for field in REQUIRED_RUN_CONTEXT_FIELDS if not _normalize_text(normalized[field])]
    return {
        "path": str(context_path),
        "fields": normalized,
        "missing_fields": missing,
    }


def validate_run_context(path: str | Path) -> dict[str, object]:
    result = extract_run_context_fields(path)
    missing = result["missing_fields"]
    if missing:
        raise ValueError(
            "RUN_CONTEXT contract violation: missing required fields "
            f"{missing} in {result['path']}"
        )
    return result


def require_run_context_for_write(run_root: str | Path) -> dict[str, object]:
    run_root_path = Path(run_root).resolve()
    if not run_root_path.exists():
        raise FileNotFoundError(f"Governed results root does not exist: {run_root_path}")
    if not run_root_path.is_dir():
        raise NotADirectoryError(f"Governed results root is not a directory: {run_root_path}")
    return validate_run_context(run_root_path / "RUN_CONTEXT.md")


def create_minimal_run_context(run_root: str | Path, metadata: dict[str, str]) -> Path:
    run_root_path = Path(run_root).resolve()
    missing = [
        field
        for field in REQUIRED_RUN_CONTEXT_FIELDS
        if not _normalize_text(metadata.get(field, ""))
    ]
    if missing:
        raise ValueError(
            "Cannot create RUN_CONTEXT.md without explicit required metadata fields: "
            f"{missing}"
        )
    lines = [
        "# RUN_CONTEXT",
        "",
        "## 1. Governed Write Contract",
        f"- run_class: `{_normalize_text(metadata['run_class'])}`",
        f"- stage_coverage: `{_normalize_text(metadata['stage_coverage'])}`",
        f"- boundary_class: `{_normalize_text(metadata['boundary_class'])}`",
        f"- lawful_resume_boundary: `{_normalize_text(metadata['lawful_resume_boundary'])}`",
        f"- upstream_authority_source: `{_normalize_text(metadata['upstream_authority_source'])}`",
        f"- created_by_script: `{_normalize_text(metadata['created_by_script'])}`",
        "",
    ]
    run_root_path.mkdir(parents=True, exist_ok=True)
    target = run_root_path / "RUN_CONTEXT.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate governed RUN_CONTEXT.md contracts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a RUN_CONTEXT.md path or run root.")
    validate_parser.add_argument("path")

    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "validate":
        payload = validate_run_context(args.path)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
