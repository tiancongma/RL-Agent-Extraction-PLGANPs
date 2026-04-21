#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from src.utils.paths import DATA_MEM_V1_DIR, DATA_RESULTS_DIR, DOCS_DIR, PROJECT_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import DATA_MEM_V1_DIR, DATA_RESULTS_DIR, DOCS_DIR, PROJECT_DIR, PROJECT_ROOT


LEGACY_RUN_ID_RE = re.compile(r"(run_\d{8}_\d{4,6}_[0-9a-f]{7}_[A-Za-z0-9_]+)")
V2_BUCKET_RE = re.compile(r"(\d{8}_[0-9a-f]{7})")
V2_CHILD_RE = re.compile(r"(\d{2,3}_[a-z0-9][a-z0-9_]*)")
STAGE_RE = re.compile(r"\b(stage\s*[0-5]|layer\s*[1-3])\b", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
DECISION_LINE_RE = re.compile(r"^\s*Decision\b\s*:?\s*(.+?)\s*$", re.IGNORECASE)
REASON_LINE_RE = re.compile(r"^\s*Reason\b\s*:?\s*(.+?)\s*$", re.IGNORECASE)
IMPACT_LINE_RE = re.compile(r"^\s*Impact\b\s*:?\s*(.+?)\s*$", re.IGNORECASE)
RUN_TYPE_LINE_RE = re.compile(r"^\s*-\s*`?([A-Za-z0-9_]+run)`?\s*$", re.IGNORECASE)
RUN_FIELD_RE = re.compile(r"^\s*-\s*([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$")
KEYWORDS = (
    "collapse",
    "mismatch",
    "missing",
    "failure",
    "fail",
    "regression",
    "root cause",
    "inflation",
    "over-count",
    "overcount",
    "error",
    "blocker",
    "drop",
    "dropped",
)
ERROR_SKIP_PREFIXES = (
    "run type:",
    "file:",
    "root:",
    "purpose:",
    "selected frozen upstream base:",
    "overlap scaffold:",
    "step ",
)
WEAK_DECISIONS = {
    "case reference",
    "date",
    "s.",
    "root:",
    "file:",
    "purpose",
    "structure",
}
WEAK_PROMPT_TITLES = {
    "asset inventory and triage",
    "conclusion",
    "immediate engineering implications",
    "new behavior",
    "output summary",
    "purpose",
    "recommendation",
    "root cause",
    "structure",
    "whether a new script is needed",
}
PROMPT_HINTS = (
    "prompt",
    "few-shot",
    "instruction",
    "query template",
    "workflow query",
    "recipe",
    "template",
    "bootstrap",
)
STATIC_PROMPTS = [
    {
        "title": "Debug bootstrap",
        "stage": "",
        "run_id": "",
        "recipe": "For debugging, query memory first with the failure symptom, key paper ID, and stage name before reading code.",
        "usage": "Example: collapse; 5GIF3D8W; stage2 parsing. Then open the top memory sources and only then inspect local scripts.",
        "tags": "debugging workflow",
        "source_file": "project/ACTIVE_PIPELINE_RUNBOOK.md",
        "source_kind": "workflow",
        "status": "active",
    },
    {
        "title": "Regression bootstrap",
        "stage": "stage2",
        "run_id": "",
        "recipe": "For regression investigation, query memory with the regression label, affected run ID, and benchmark scope before acting.",
        "usage": "Example: regression; DOE; dev15. Review the recalled runs, decisions, and failure patterns before rerunning anything.",
        "tags": "regression workflow",
        "source_file": "project/ACTIVE_PIPELINE_RUNBOOK.md",
        "source_kind": "workflow",
        "status": "active",
    },
    {
        "title": "Run compare bootstrap",
        "stage": "",
        "run_id": "",
        "recipe": "For run comparison, query memory with both the run theme and lineage terms so prior child runs and benchmark notes surface together.",
        "usage": "Example: run lineage; blocker gate; dev15 current merged benchmark. Then inspect RUN_CONTEXT and lineage sources from the top hits.",
        "tags": "run comparison workflow",
        "source_file": "project/ACTIVE_PIPELINE_RUNBOOK.md",
        "source_kind": "workflow",
        "status": "active",
    },
    {
        "title": "Pipeline change bootstrap",
        "stage": "",
        "run_id": "",
        "recipe": "For pipeline modification, query memory for the stage plus the intended behavior change to recover prior decisions and guardrails first.",
        "usage": "Example: family variant; table-first; stage5 descendant filter; stage2 parsing.",
        "tags": "pipeline modification workflow",
        "source_file": "project/ACTIVE_PIPELINE_RUNBOOK.md",
        "source_kind": "workflow",
        "status": "active",
    },
    {
        "title": "GT mismatch bootstrap",
        "stage": "stage5",
        "run_id": "",
        "recipe": "For GT mismatch analysis, query memory with the paper key or mismatch phrase before opening comparison artifacts.",
        "usage": "Example: identity mismatch; BB3JUVW7; family variant. Use memory hits to recover prior mismatch explanations and fix history.",
        "tags": "gt mismatch workflow",
        "source_file": "project/ACTIVE_PIPELINE_RUNBOOK.md",
        "source_kind": "workflow",
        "status": "active",
    },
    {
        "title": "Lineage bootstrap",
        "stage": "",
        "run_id": "",
        "recipe": "For lineage tracing, query memory with run lineage terms first so parent-child run structure is recalled before filesystem traversal.",
        "usage": "Example: run lineage; dev15 deterministic refresh; targeted5 stage2 regression.",
        "tags": "lineage tracing workflow",
        "source_file": "project/ACTIVE_PIPELINE_RUNBOOK.md",
        "source_kind": "workflow",
        "status": "active",
    },
]


@dataclass(frozen=True)
class Section:
    title: str
    body: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mem_v1 TSV registries from governed repo sources.")
    parser.add_argument("--mem-dir", type=Path, default=DATA_MEM_V1_DIR, help="Memory directory. Default: data/mem/v1")
    parser.add_argument("--force", action="store_true", help="Accepted for compatibility; rebuilds are already explicit and logged.")
    parser.add_argument("--init-only", action="store_true", help="Create headers/schema only and stop.")
    return parser.parse_args()


def load_schema(mem_dir: Path) -> dict:
    schema_path = mem_dir / "sch.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema manifest: {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_rows(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: str(row.get(key, "")) for key in headers})


def has_data_rows(path: Path) -> bool:
    return bool(load_rows(path))


def repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def clean_text(value: str) -> str:
    text = str(value or "").replace("\ufeff", " ").strip()
    return re.sub(r"\s+", " ", text)


def first_text(lines: Iterable[str]) -> str:
    for line in lines:
        text = clean_text(str(line).lstrip("-* "))
        if text:
            return text
    return ""


def first_meaningful_text(lines: Iterable[str]) -> str:
    for line in lines:
        text = clean_text(line)
        if not text:
            continue
        if text.startswith("#"):
            continue
        if text.startswith("##"):
            continue
        if text.startswith("```"):
            continue
        if text.lower() == "run_context":
            continue
        if text.startswith("`run_"):
            continue
        if RUN_FIELD_RE.match(text):
            continue
        return text.lstrip("-* ")
    return ""


def infer_stage(*values: str) -> str:
    blob = " ".join(clean_text(value) for value in values if clean_text(value))
    match = STAGE_RE.search(blob)
    if not match:
        return ""
    return match.group(1).lower().replace(" ", "")


def find_run_id(*values: str) -> str:
    blob = " ".join(str(value or "") for value in values)
    match = LEGACY_RUN_ID_RE.search(blob)
    return match.group(1) if match else ""


def is_supported_run_id(value: str) -> bool:
    text = clean_text(value).lower()
    return bool(
        LEGACY_RUN_ID_RE.fullmatch(text)
        or V2_BUCKET_RE.fullmatch(text)
        or V2_CHILD_RE.fullmatch(text)
    )


def extract_supported_run_id(value: str) -> str:
    text = clean_text(value)
    if not text:
        return ""
    for candidate in re.findall(r"`([^`]+)`", text):
        if is_supported_run_id(candidate):
            return clean_text(candidate).lower()
    if is_supported_run_id(text):
        return text.lower()
    return ""


def parse_run_id_from_context(text: str, path: Path) -> str:
    lines = text.splitlines()
    for line in lines:
        match = RUN_FIELD_RE.match(line)
        if match and clean_text(match.group(1)).lower() == "run_id":
            run_id = extract_supported_run_id(match.group(2))
            if run_id:
                return run_id
    in_run_id_section = False
    for line in lines:
        if line.strip().startswith("## "):
            heading = clean_text(line.lstrip("# ")).lower()
            in_run_id_section = heading.endswith("run id") or heading == "run id"
            continue
        if not in_run_id_section:
            continue
        if not clean_text(line):
            continue
        run_id = extract_supported_run_id(line.lstrip("-* "))
        if run_id:
            return run_id
    if is_supported_run_id(path.parent.name):
        return path.parent.name.lower()
    return ""


def split_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    current_title = ""
    current_body: list[str] = []
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            if current_title or current_body:
                sections.append(Section(current_title, current_body))
            current_title = clean_text(match.group(2))
            current_body = []
        else:
            current_body.append(line)
    if current_title or current_body:
        sections.append(Section(current_title, current_body))
    return sections


def collect_sources() -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for path in sorted((DOCS_DIR / "snapshots").glob("*.md")):
        sources.append(("snapshot", path))
    for path in sorted((DOCS_DIR / "methods").glob("*.md")):
        if path.name == "mem_v1_audit.md":
            continue
        sources.append(("method", path))
    for path in sorted(DATA_RESULTS_DIR.rglob("RUN_CONTEXT.md")):
        sources.append(("run_context", path))
    for path in sorted(PROJECT_DIR.glob("*.md")):
        sources.append(("project", path))
    return sources


def parse_run_type(text: str) -> str:
    lines = text.splitlines()
    for line in lines:
        match = RUN_FIELD_RE.match(line)
        if match and clean_text(match.group(1)).lower() == "run_type":
            return clean_text(match.group(2))
    in_section = False
    for line in lines:
        if line.strip().startswith("## "):
            in_section = "run type" in line.lower()
            continue
        if not in_section:
            continue
        match = RUN_TYPE_LINE_RE.match(line)
        if match:
            return clean_text(match.group(1))
    return ""


def parse_purpose(text: str) -> str:
    lines = text.splitlines()
    field_hits: list[str] = []
    for line in lines:
        match = RUN_FIELD_RE.match(line)
        if not match:
            continue
        key = clean_text(match.group(1)).lower()
        value = clean_text(match.group(2))
        if key in {"run_purpose", "purpose", "why_this_run_exists"} and value:
            field_hits.append(value)
    if field_hits:
        return " | ".join(field_hits[:3])
    in_section = False
    bullets: list[str] = []
    for line in lines:
        if line.strip().startswith("## "):
            lower = line.lower()
            in_section = "purpose" in lower or "why this run exists" in lower
            continue
        if not in_section:
            continue
        if line.strip().startswith("- "):
            bullets.append(clean_text(line.lstrip("- ")))
        elif bullets and not line.strip():
            break
    return " | ".join(bullets[:3])


def parent_run_from_path(path: Path, current_run_id: str) -> str:
    ancestors = list(path.parents)[1:]
    ancestor_contexts: list[Path] = []
    for ancestor in ancestors:
        candidate = ancestor / "RUN_CONTEXT.md"
        if candidate.exists():
            ancestor_contexts.append(candidate)
    for ancestor_context in ancestor_contexts:
        ancestor_run_id = parse_run_id_from_context(ancestor_context.read_text(encoding="utf-8", errors="replace"), ancestor_context)
        if not ancestor_run_id:
            raise ValueError(f"Ancestor RUN_CONTEXT lacks explicit run_id: {repo_rel(ancestor_context)}")
        if ancestor_run_id != current_run_id:
            return ancestor_run_id
    for ancestor in ancestors:
        ancestor_name = clean_text(ancestor.name).lower()
        if is_supported_run_id(ancestor_name) and ancestor_name != current_run_id:
            return ancestor_name
    rel_parts = path.resolve().relative_to(DATA_RESULTS_DIR.resolve()).parts
    if len(rel_parts) >= 3 and V2_BUCKET_RE.fullmatch(rel_parts[0]) and V2_CHILD_RE.fullmatch(rel_parts[1]):
        raise ValueError(f"Missing explicit bucket parent RUN_CONTEXT: {repo_rel(path)}")
    if "lineage" in rel_parts and "children" in rel_parts:
        raise ValueError(f"Missing explicit lineage parent RUN_CONTEXT: {repo_rel(path)}")
    return ""


def build_run_rows(sources: list[tuple[str, Path]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    run_rows: list[dict[str, str]] = []
    lin_rows: list[dict[str, str]] = []
    for source_kind, path in sources:
        if source_kind != "run_context":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        run_id = parse_run_id_from_context(text, path)
        if not run_id:
            continue
        run_type = parse_run_type(text)
        purpose = parse_purpose(text)
        summary = purpose or first_meaningful_text(text.splitlines()) or first_text(text.splitlines())
        parent_run = parent_run_from_path(path, run_id)
        run_rows.append(
            {
                "run_mem_id": "",
                "run_id": run_id,
                "run_type": run_type,
                "stage": infer_stage(summary, run_id),
                "parent_run": parent_run,
                "purpose": purpose,
                "summary": summary,
                "source_file": repo_rel(path),
                "source_kind": source_kind,
                "status": "active",
            }
        )
        if parent_run:
            lin_rows.append(
                {
                    "lin_id": "",
                    "parent_run": parent_run,
                    "child_run": run_id,
                    "relation": "lineage_child",
                    "source_file": repo_rel(path),
                    "note": "derived_from_run_context_path",
                }
            )
    known_run_ids = {row["run_id"] for row in run_rows if row.get("run_id")}
    synthetic_parent_rows: list[dict[str, str]] = []
    for lin_row in lin_rows:
        parent_run = lin_row["parent_run"]
        if parent_run in known_run_ids:
            continue
        if not V2_BUCKET_RE.fullmatch(parent_run):
            raise ValueError(f"Lineage parent missing explicit run row: {parent_run}")
        synthetic_parent_rows.append(
            {
                "run_mem_id": "",
                "run_id": parent_run,
                "run_type": "v2_bucket_parent",
                "stage": "",
                "parent_run": "",
                "purpose": "",
                "summary": "Synthetic bucket parent reconstructed from explicit v2 bucket/child lineage path because bucket root RUN_CONTEXT.md is absent.",
                "source_file": lin_row["source_file"],
                "source_kind": "path_contract",
                "status": "active",
            }
        )
        known_run_ids.add(parent_run)
    run_rows.extend(synthetic_parent_rows)
    return run_rows, lin_rows


def extract_decision_rows(source_kind: str, path: Path, text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for section in split_sections(text):
        title_lower = section.title.lower()
        decision = ""
        reason = ""
        impact = ""
        body = [clean_text(line) for line in section.body if clean_text(line)]
        if "decision" in title_lower and body:
            decision = first_text(body)
        for line in body:
            decision_match = DECISION_LINE_RE.match(line)
            if decision_match:
                decision = clean_text(decision_match.group(1))
            reason_match = REASON_LINE_RE.match(line)
            if reason_match and not reason:
                reason = clean_text(reason_match.group(1))
            impact_match = IMPACT_LINE_RE.match(line)
            if impact_match and not impact:
                impact = clean_text(impact_match.group(1))
        if not decision and title_lower.startswith("decision:"):
            decision = clean_text(section.title.split(":", 1)[1])
        if decision.lower() in WEAK_DECISIONS and title_lower.startswith("decision:"):
            decision = clean_text(section.title.split(":", 1)[1])
        if not decision:
            continue
        if "decision" not in title_lower and not any(DECISION_LINE_RE.match(line) for line in body):
            continue
        if decision.lower() in WEAK_DECISIONS or len(decision) < 20:
            continue
        rows.append(
            {
                "dec_id": "",
                "stage": infer_stage(section.title, decision, reason, impact),
                "run_id": find_run_id(text, section.title, decision),
                "title": section.title or decision[:80],
                "decision": decision,
                "reason": reason,
                "impact": impact,
                "tags": source_kind,
                "source_file": repo_rel(path),
                "source_kind": source_kind,
                "status": "active",
            }
        )
    return rows


def classify_error_signature(line: str) -> str:
    lowered = line.lower()
    hits = [token.replace(" ", "_").replace("-", "_") for token in KEYWORDS if token in lowered]
    if hits:
        return "_".join(hits[:2])
    words = re.findall(r"[a-z0-9]+", lowered)
    return "_".join(words[:3])[:32]


def extract_error_rows(source_kind: str, path: Path, text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    in_code = False
    recent_heading = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        heading_match = HEADING_RE.match(raw_line)
        if heading_match:
            recent_heading = clean_text(heading_match.group(2))
            continue
        line = clean_text(stripped.lstrip("-* "))
        if not line or len(line) < 20:
            continue
        if raw_line.strip().startswith(("1.", "2.", "3.", "4.", "5.")) and ".py" in raw_line:
            continue
        if ".py" in line or "powershell" in line.lower():
            continue
        if ":\\" in line or "/data/results/" in line.lower():
            continue
        if lowered := line.lower():
            if any(lowered.startswith(prefix) for prefix in ERROR_SKIP_PREFIXES):
                continue
        else:
            continue
        if not any(keyword in lowered for keyword in KEYWORDS):
            continue
        rows.append(
            {
                "err_id": "",
                "stage": infer_stage(recent_heading, line),
                "run_id": find_run_id(str(path), line),
                "err_sig": classify_error_signature(line),
                "symptom": line,
                "cause": recent_heading,
                "fix": "",
                "tags": source_kind,
                "source_file": repo_rel(path),
                "source_kind": source_kind,
                "status": "active",
            }
        )
    return rows


def extract_prompt_rows(source_kind: str, path: Path, text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for section in split_sections(text):
        body = [clean_text(line) for line in section.body if clean_text(line)]
        body_text = " ".join(body)
        title_lower = section.title.lower()
        body_lower = body_text.lower()
        if title_lower in WEAK_PROMPT_TITLES:
            continue
        if not any(
            re.search(r"\b" + re.escape(hint) + r"\b", title_lower)
            or re.search(r"\b" + re.escape(hint) + r"\b", body_lower)
            for hint in PROMPT_HINTS
        ):
            continue
        if source_kind == "run_context" and not any(
            re.search(r"\b" + re.escape(hint) + r"\b", title_lower)
            for hint in PROMPT_HINTS
        ):
            continue
        recipe = first_text(body) or section.title
        if len(recipe) < 20:
            continue
        rows.append(
            {
                "prm_id": "",
                "stage": infer_stage(section.title, body_text),
                "run_id": find_run_id(body_text),
                "title": section.title or body_text[:80],
                "recipe": recipe,
                "usage": body_text[:240],
                "tags": source_kind,
                "source_file": repo_rel(path),
                "source_kind": source_kind,
                "status": "active",
            }
        )
    return rows


def dedupe_rows(rows: list[dict[str, str]], key_fields: list[str]) -> list[dict[str, str]]:
    seen: set[tuple[str, ...]] = set()
    deduped: list[dict[str, str]] = []
    for row in rows:
        key = tuple(clean_text(row.get(field, "")) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def normalize_symptom(value: str) -> str:
    text = clean_text(value).lower()
    text = text.replace("`", "")
    text = re.sub(r"run_\d{8}_\d{4}_[0-9a-f]{7}_[a-z0-9_]+", "run_id", text)
    text = re.sub(r"\b\d+\b", "n", text)
    return text


def aggregate_error_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("err_sig", ""), normalize_symptom(row.get("symptom", "")))
        grouped.setdefault(key, []).append(row)
    aggregated: list[dict[str, str]] = []
    for _, group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        group = sorted(group, key=lambda row: (row.get("run_id", ""), row.get("source_file", ""), row.get("symptom", "")))
        base = dict(group[0])
        if len(group) > 1:
            tags = clean_text(base.get("tags", ""))
            base["tags"] = f"{tags} seen:{len(group)}".strip()
        aggregated.append(base)
    return aggregated


def existing_id_map(existing_rows: list[dict[str, str]], id_field: str, key_fields: list[str]) -> dict[tuple[str, ...], str]:
    mapping: dict[tuple[str, ...], str] = {}
    for row in existing_rows:
        key = tuple(clean_text(row.get(field, "")) for field in key_fields)
        if key and row.get(id_field):
            mapping[key] = row[id_field]
    return mapping


def next_id(prefix: str, taken: set[str]) -> str:
    max_seen = 0
    for value in taken:
        if value.startswith(prefix):
            suffix = value[len(prefix):]
            if suffix.isdigit():
                max_seen = max(max_seen, int(suffix))
    candidate = f"{prefix}{max_seen + 1:03d}"
    taken.add(candidate)
    return candidate


def assign_ids(rows: list[dict[str, str]], existing_rows: list[dict[str, str]], id_field: str, prefix: str, key_fields: list[str]) -> list[dict[str, str]]:
    mapping = existing_id_map(existing_rows, id_field, key_fields)
    taken = {value for value in mapping.values() if value}
    assigned: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: tuple(clean_text(item.get(field, "")) for field in key_fields)):
        key = tuple(clean_text(row.get(field, "")) for field in key_fields)
        copy = dict(row)
        copy[id_field] = mapping.get(key) or next_id(prefix, taken)
        assigned.append(copy)
    return assigned


def build_idx_rows(
    run_rows: list[dict[str, str]],
    lin_rows: list[dict[str, str]],
    dec_rows: list[dict[str, str]],
    err_rows: list[dict[str, str]],
    prm_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in run_rows:
        rows.append(
            {
                "mem_id": "",
                "mem_type": "run",
                "ref_id": row["run_mem_id"],
                "stage": row["stage"],
                "run_id": row["run_id"],
                "title": row["run_id"],
                "summary": row["summary"],
                "tags": row["run_type"],
                "source_file": row["source_file"],
                "source_kind": row["source_kind"],
                "status": row["status"],
            }
        )
    for row in lin_rows:
        rows.append(
            {
                "mem_id": "",
                "mem_type": "lineage",
                "ref_id": row["lin_id"],
                "stage": "",
                "run_id": row["child_run"],
                "title": f"{row['child_run']} <- {row['parent_run']}",
                "summary": row["note"],
                "tags": row["relation"],
                "source_file": row["source_file"],
                "source_kind": "run_context",
                "status": "active",
            }
        )
    for row in dec_rows:
        rows.append(
            {
                "mem_id": "",
                "mem_type": "decision",
                "ref_id": row["dec_id"],
                "stage": row["stage"],
                "run_id": row["run_id"],
                "title": row["title"],
                "summary": row["decision"],
                "tags": row["tags"],
                "source_file": row["source_file"],
                "source_kind": row["source_kind"],
                "status": row["status"],
            }
        )
    for row in err_rows:
        rows.append(
            {
                "mem_id": "",
                "mem_type": "error",
                "ref_id": row["err_id"],
                "stage": row["stage"],
                "run_id": row["run_id"],
                "title": row["err_sig"],
                "summary": row["symptom"],
                "tags": row["tags"],
                "source_file": row["source_file"],
                "source_kind": row["source_kind"],
                "status": row["status"],
            }
        )
    for row in prm_rows:
        rows.append(
            {
                "mem_id": "",
                "mem_type": "prompt",
                "ref_id": row["prm_id"],
                "stage": row["stage"],
                "run_id": row["run_id"],
                "title": row["title"],
                "summary": row["recipe"],
                "tags": row["tags"],
                "source_file": row["source_file"],
                "source_kind": row["source_kind"],
                "status": row["status"],
            }
        )
    return rows


def initialize_headers(mem_dir: Path, schema: dict) -> None:
    mem_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in schema["tables"].items():
        path = mem_dir / name
        if not path.exists():
            write_rows(path, spec["headers"], [])
    print(f"mem_dir={mem_dir}")
    print("status=initialized")


def main() -> int:
    args = parse_args()
    mem_dir = args.mem_dir.resolve()
    schema = load_schema(mem_dir)
    initialize_headers(mem_dir, schema)
    if args.init_only:
        return 0
    populated = [name for name in schema["tables"] if has_data_rows(mem_dir / name)]
    print(f"mode={'rebuild' if populated else 'initial_build'}")
    if populated:
        print("refreshing=" + ",".join(sorted(populated)))
    sources = collect_sources()
    run_rows, lin_rows = build_run_rows(sources)
    dec_rows: list[dict[str, str]] = []
    err_rows: list[dict[str, str]] = []
    prm_rows: list[dict[str, str]] = []
    for source_kind, path in sources:
        text = path.read_text(encoding="utf-8", errors="replace")
        dec_rows.extend(extract_decision_rows(source_kind, path, text))
        err_rows.extend(extract_error_rows(source_kind, path, text))
        prm_rows.extend(extract_prompt_rows(source_kind, path, text))
    prm_rows.extend(STATIC_PROMPTS)

    run_rows = dedupe_rows(run_rows, ["run_id", "source_file"])
    lin_rows = dedupe_rows(lin_rows, ["parent_run", "child_run", "relation"])
    dec_rows = dedupe_rows(dec_rows, ["source_file", "title", "decision"])
    err_rows = dedupe_rows(err_rows, ["source_file", "err_sig", "symptom"])
    prm_rows = dedupe_rows(prm_rows, ["source_file", "title", "recipe"])
    err_rows = aggregate_error_rows(err_rows)

    run_rows = assign_ids(run_rows, load_rows(mem_dir / "run.tsv"), "run_mem_id", schema["tables"]["run.tsv"]["id_prefix"], ["run_id", "source_file"])
    lin_rows = assign_ids(lin_rows, load_rows(mem_dir / "lin.tsv"), "lin_id", schema["tables"]["lin.tsv"]["id_prefix"], ["parent_run", "child_run", "relation"])
    dec_rows = assign_ids(dec_rows, load_rows(mem_dir / "dec.tsv"), "dec_id", schema["tables"]["dec.tsv"]["id_prefix"], ["source_file", "title", "decision"])
    err_rows = assign_ids(err_rows, load_rows(mem_dir / "err.tsv"), "err_id", schema["tables"]["err.tsv"]["id_prefix"], ["source_file", "err_sig", "symptom"])
    prm_rows = assign_ids(prm_rows, load_rows(mem_dir / "prm.tsv"), "prm_id", schema["tables"]["prm.tsv"]["id_prefix"], ["source_file", "title", "recipe"])

    idx_rows = build_idx_rows(run_rows, lin_rows, dec_rows, err_rows, prm_rows)
    idx_rows = dedupe_rows(idx_rows, ["mem_type", "ref_id"])
    idx_rows = assign_ids(idx_rows, load_rows(mem_dir / "idx.tsv"), "mem_id", schema["tables"]["idx.tsv"]["id_prefix"], ["mem_type", "ref_id"])

    write_rows(mem_dir / "run.tsv", schema["tables"]["run.tsv"]["headers"], run_rows)
    write_rows(mem_dir / "lin.tsv", schema["tables"]["lin.tsv"]["headers"], lin_rows)
    write_rows(mem_dir / "dec.tsv", schema["tables"]["dec.tsv"]["headers"], dec_rows)
    write_rows(mem_dir / "err.tsv", schema["tables"]["err.tsv"]["headers"], err_rows)
    write_rows(mem_dir / "prm.tsv", schema["tables"]["prm.tsv"]["headers"], prm_rows)
    write_rows(mem_dir / "idx.tsv", schema["tables"]["idx.tsv"]["headers"], idx_rows)

    print(f"sources={len(sources)}")
    print(f"runs={len(run_rows)}")
    print(f"lineage={len(lin_rows)}")
    print(f"decisions={len(dec_rows)}")
    print(f"errors={len(err_rows)}")
    print(f"prompts={len(prm_rows)}")
    print(f"index={len(idx_rows)}")
    print("status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
