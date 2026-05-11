#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    from src.utils.active_data_source import resolve_run_context
    from src.utils.paths import DATA_MEM_V1_DIR, DOCS_DIR, PROJECT_ROOT
    from src.utils.query_mem_v1 import query_memory
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.active_data_source import resolve_run_context
    from src.utils.paths import DATA_MEM_V1_DIR, DOCS_DIR, PROJECT_ROOT
    from src.utils.query_mem_v1 import query_memory


REPAIR_INDEX_PATH = DOCS_DIR / "repair_index" / "success_pattern_index_v1.tsv"
DEFAULT_MEMORY_LIMIT = 8
DEFAULT_PATTERN_LIMIT = 8
DEFAULT_BOUNDARY_LIMIT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only repair intake for the governed repair-pattern-guided regression workflow."
    )
    parser.add_argument("--paper-key", default="", help="Optional paper key to focus the intake.")
    parser.add_argument("--failure-phrase", default="", help="Optional failure phrase for repair-index and memory lookup.")
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=None,
        help="Optional explicit baseline run directory. Defaults to data/results/ACTIVE_RUN.json authority.",
    )
    parser.add_argument(
        "--write-tsv",
        action="store_true",
        help="Optionally write analysis/repair_intake_<timestamp>.tsv under the resolved baseline run directory.",
    )
    return parser.parse_args()


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def normalize(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize(value))


def build_search_terms(*values: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for value in values:
        value = str(value or "").strip()
        if not value:
            continue
        variants = [value, normalize(value)]
        variants.extend(tokenize(value))
        for item in variants:
            item = str(item or "").strip()
            if not item or item in seen:
                continue
            seen.add(item)
            terms.append(item)
    if not terms:
        terms.extend(["regression", "capability restoration", "missing rows"])
    return terms


def score_pattern(row: dict[str, str], search_terms: list[str]) -> int:
    corpus_fields = [
        "pattern_id",
        "title",
        "failure_type",
        "paper_class",
        "trigger_signature",
        "repair_unit_type",
        "notes",
        "adoption_status",
        "benchmark_effect",
        "no_regression_scope",
        "linked_governance_changes",
    ]
    corpus = " || ".join(normalize(row.get(field, "")) for field in corpus_fields)
    score = 0
    for term in search_terms:
        normalized_term = normalize(term)
        if not normalized_term:
            continue
        if normalized_term == normalize(row.get("pattern_id", "")):
            score += 120
        if normalized_term in corpus:
            score += 35
        for token in tokenize(normalized_term):
            if token and token in corpus:
                score += 8
    if row.get("activation_evidence_strength", "") == "explicit_governed_activation":
        score += 10
    if row.get("adoption_status", "").startswith("validated"):
        score += 8
    return score


def select_patterns(rows: list[dict[str, str]], search_terms: list[str], limit: int) -> list[dict[str, str]]:
    ranked: list[tuple[int, dict[str, str]]] = []
    for row in rows:
        score = score_pattern(row, search_terms)
        if score > 0:
            ranked.append((score, row))
    ranked.sort(
        key=lambda item: (
            -item[0],
            item[1].get("activation_evidence_strength", ""),
            item[1].get("pattern_id", ""),
        )
    )
    return [row for _, row in ranked[:limit]]


def print_step_header(title: str) -> None:
    print(title)


def summarize_run_context(run_context_path: Path) -> tuple[list[str], list[str]]:
    if not run_context_path.exists():
        return (["RUN_CONTEXT.md missing"], [])
    lines = run_context_path.read_text(encoding="utf-8", errors="replace").splitlines()
    summary: list[str] = []
    boundary_lines: list[str] = []
    in_feature_section = False
    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()
        if stripped.startswith("## "):
            in_feature_section = stripped == "## Feature Unit Activation"
        if stripped.startswith("## 2. Run Type") or stripped.startswith("## 3. Purpose") or stripped.startswith("## 4. Starting Inputs"):
            summary.append(stripped)
            continue
        if summary and stripped.startswith("- `") and len(summary) < 10:
            summary.append(stripped)
        if any(
            marker in lowered
            for marker in (
                "boundary",
                "resume",
                "diagnostic-only",
                "benchmark-valid",
                "benchmark_mode",
                "run_activation_gate",
                "feature_activation_report_path",
                "source_mode",
                "historical only",
                "partially restored",
            )
        ):
            boundary_lines.append(stripped)
        if in_feature_section and stripped.startswith("- "):
            boundary_lines.append(stripped)
    return (summary[:8], boundary_lines[:DEFAULT_BOUNDARY_LIMIT])


def resolve_feature_activation_path(run_dir: Path, boundary_lines: list[str]) -> Path | None:
    for line in boundary_lines:
        match = re.search(r"`([^`]*feature_activation_report_v1\.tsv)`", line)
        if match:
            candidate = (PROJECT_ROOT / match.group(1)).resolve()
            if candidate.exists():
                return candidate
    fallback = run_dir / "analysis" / "feature_activation_report_v1.tsv"
    if fallback.exists():
        return fallback
    return None


def read_feature_activation_rows(path: Path | None, limit: int = 8) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    rows = load_tsv(path)
    interesting: list[dict[str, str]] = []
    for row in rows:
        activation_state = row.get("activation_state", "") or row.get("activation_status", "")
        if activation_state and activation_state.lower() != "not_applicable":
            interesting.append(row)
    return interesting[:limit]


def classify_activation_status(patterns: list[dict[str, str]], feature_rows: list[dict[str, str]], boundary_lines: list[str]) -> str:
    statuses = {row.get("activation_evidence_strength", "") for row in patterns}
    boundary_text = " ".join(boundary_lines).lower()
    feature_text = " ".join(
        f"{row.get('feature_id','')} {row.get('activation_state','')} {row.get('notes','')}" for row in feature_rows
    ).lower()
    if not feature_rows and not boundary_lines:
        return "historical only"
    if "historical only" in boundary_text:
        return "historical only"
    if "partially restored" in boundary_text:
        return "partially restored"
    if "explicit_governed_activation" in statuses and "missing" not in feature_text:
        return "active"
    if statuses or "partial" in feature_text or "missing" in feature_text:
        return "partially restored"
    return "historical only"


def write_analysis_tsv(
    *,
    run_dir: Path,
    paper_key: str,
    failure_phrase: str,
    patterns: list[dict[str, str]],
    memory_hits: list[dict[str, str]],
    activation_status: str,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_path = analysis_dir / f"repair_intake_{timestamp}.tsv"
    fieldnames = [
        "section",
        "rank",
        "paper_key",
        "failure_phrase",
        "pattern_id",
        "mem_id",
        "title",
        "status",
        "source",
        "summary",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for idx, row in enumerate(patterns, start=1):
            writer.writerow(
                {
                    "section": "repair_index",
                    "rank": idx,
                    "paper_key": paper_key,
                    "failure_phrase": failure_phrase,
                    "pattern_id": row.get("pattern_id", ""),
                    "mem_id": "",
                    "title": row.get("title", ""),
                    "status": row.get("activation_evidence_strength", ""),
                    "source": row.get("evidence_run_context_path", ""),
                    "summary": row.get("notes", ""),
                }
            )
        for idx, row in enumerate(memory_hits, start=1):
            writer.writerow(
                {
                    "section": "memory",
                    "rank": idx,
                    "paper_key": paper_key,
                    "failure_phrase": failure_phrase,
                    "pattern_id": "",
                    "mem_id": row.get("mem_id", ""),
                    "title": row.get("title", ""),
                    "status": activation_status,
                    "source": row.get("source_file", ""),
                    "summary": row.get("summary", ""),
                }
            )
    return output_path


def main() -> int:
    args = parse_args()
    search_terms = build_search_terms(args.paper_key, args.failure_phrase)

    print_step_header("STEP 1: Repair Index Lookup")
    repair_rows = load_tsv(REPAIR_INDEX_PATH)
    pattern_hits = select_patterns(repair_rows, search_terms, DEFAULT_PATTERN_LIMIT)
    print(f"repair_index={REPAIR_INDEX_PATH}")
    if pattern_hits:
        for idx, row in enumerate(pattern_hits, start=1):
            print(
                f"pattern_{idx}={row.get('pattern_id','')} | {row.get('title','')} | "
                f"activation_evidence_strength={row.get('activation_evidence_strength','')} | "
                f"adoption_status={row.get('adoption_status','')}"
            )
    else:
        print("pattern_hits=none")

    print_step_header("STEP 2: Memory Query")
    memory_query = " ".join(part for part in [args.paper_key, args.failure_phrase, "regression repair"] if part)
    if not memory_query:
        memory_query = "regression repair"
    print(f"memory_dir={DATA_MEM_V1_DIR}")
    print(f"memory_query={memory_query}")
    memory_hits = query_memory(mem_dir=DATA_MEM_V1_DIR.resolve(), query=memory_query, limit=DEFAULT_MEMORY_LIMIT)
    if memory_hits:
        for idx, row in enumerate(memory_hits, start=1):
            print(
                f"memory_{idx}={row.get('mem_id','')} | {row.get('mem_type','')} | "
                f"{row.get('title','')} | {row.get('source_file','')}"
            )
    else:
        print("memory_hits=none")

    print_step_header("STEP 3: Baseline + Run Evidence")
    run_context = resolve_run_context(explicit_run_dir=args.baseline_run_dir)
    run_dir = Path(run_context["run_dir"]).resolve()
    run_context_path = run_dir / "RUN_CONTEXT.md"
    print(f"resolved_run_dir={run_dir}")
    print(f"resolved_run_id={run_context['run_id']}")
    print(f"resolution_source={run_context['resolution_source']}")
    if run_context.get("pointer_path"):
        print(f"active_run_pointer={run_context['pointer_path']}")
    if run_context.get("pointer_payload"):
        terminal_files = (run_context["pointer_payload"] or {}).get("authoritative_terminal_files", {})
        for key, value in sorted(terminal_files.items()):
            print(f"source_file[{key}]={value}")
    run_summary, boundary_lines = summarize_run_context(run_context_path)
    print(f"run_context_path={run_context_path}")
    for line in run_summary:
        print(f"run_context_summary={line}")
    feature_activation_path = resolve_feature_activation_path(run_dir, boundary_lines)
    print(f"feature_activation_path={feature_activation_path if feature_activation_path else 'missing'}")
    feature_rows = read_feature_activation_rows(feature_activation_path)
    for row in feature_rows:
        print(
            f"feature_activation={row.get('feature_id','')} | "
            f"{row.get('activation_state', row.get('activation_status',''))} | "
            f"{row.get('evidence_path','')}"
        )
    if not feature_rows:
        print("feature_activation=none")
    for line in boundary_lines:
        print(f"boundary_governance={line}")
    if not boundary_lines:
        print("boundary_governance=none")

    print_step_header("STEP 4: Suggested Repair Direction")
    activation_status = classify_activation_status(pattern_hits, feature_rows, boundary_lines)
    print(f"activation_status={activation_status}")
    if pattern_hits:
        candidate_ids = ", ".join(row.get("pattern_id", "") for row in pattern_hits)
        print(f"candidate_patterns={candidate_ids}")
    else:
        print("candidate_patterns=none")
    gaps: list[str] = []
    if not pattern_hits:
        gaps.append("no_repair_index_match")
    if not memory_hits:
        gaps.append("no_memory_hits")
    if feature_activation_path is None:
        gaps.append("missing_feature_activation_report")
    if activation_status != "active":
        gaps.append(f"activation_status={activation_status}")
    print(f"gaps={'; '.join(gaps) if gaps else 'none'}")

    if args.write_tsv:
        output_path = write_analysis_tsv(
            run_dir=run_dir,
            paper_key=args.paper_key,
            failure_phrase=args.failure_phrase,
            patterns=pattern_hits,
            memory_hits=memory_hits,
            activation_status=activation_status,
        )
        print(f"analysis_tsv={output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
