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


TYPE_MAP = {
    "all": "",
    "run": "run",
    "runs": "run",
    "lineage": "lineage",
    "lin": "lineage",
    "decision": "decision",
    "dec": "decision",
    "error": "error",
    "err": "error",
    "prompt": "prompt",
    "prm": "prompt",
}
QUERY_EXPANSIONS = {
    "family variant": ["family variant", "family_variant", "variant", "variant_formulation"],
    "doe": ["doe", "numbered doe", "numbered_doe", "design table", "enumeration"],
    "table-first": ["table-first", "table first", "table_first", "packing", "evidence packing"],
    "identity mismatch": ["identity mismatch", "mismatch", "missing_in_system", "identity compare"],
    "stage2 parsing": ["stage2 parsing", "fixparse", "parse", "parsing", "input assembly", "fixparse input"],
    "run lineage": ["run lineage", "lineage", "children", "parent run", "lineage_child"],
}
TYPE_HINTS = {
    "lineage": {"lineage": 60, "run": 4},
    "run": {"run": 12, "lineage": 2},
    "decision": {"decision": 18},
    "prompt": {"prompt": 18},
    "error": {"error": 18},
}
STOPWORDS = {"the", "and", "for", "with", "from", "into", "this", "that"}
GENERIC_TOKENS = {"run", "stage", "stage2", "stage3", "stage4", "stage5"}
GENERIC_PROMPT_TITLES = {"conclusion", "new behavior", "output summary", "purpose", "recommendation", "structure"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query mem_v1 registry tables.")
    parser.add_argument("--mem-dir", type=Path, default=DATA_MEM_V1_DIR, help="Memory directory. Default: data/mem/v1")
    parser.add_argument("--query", default="", help="Case-insensitive text query.")
    parser.add_argument("--type", default="all", help="Memory type: all, run, lineage, decision, error, prompt.")
    parser.add_argument("--stage", default="", help="Optional stage filter.")
    parser.add_argument("--run", default="", help="Optional run_id filter.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum rows to return.")
    parser.add_argument("--format", choices=("table", "json"), default="table", help="Output format.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def normalize(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", normalize(value)) if token and token not in STOPWORDS]


def expand_terms(query: str) -> list[str]:
    query = normalize(query)
    terms = [query] if query else []
    for key, expanded in QUERY_EXPANSIONS.items():
        if query == normalize(key):
            terms.extend(normalize(item) for item in expanded)
    if not terms and query:
        terms.append(query)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped


def row_corpus(row: dict[str, str]) -> dict[str, str]:
    return {
        "title": normalize(row.get("title", "")),
        "summary": normalize(row.get("summary", "")),
        "tags": normalize(row.get("tags", "")),
        "source": normalize(row.get("source_file", "")),
        "source_kind": normalize(row.get("source_kind", "")),
        "stage": normalize(row.get("stage", "")),
        "run": normalize(row.get("run_id", "")),
        "type": normalize(row.get("mem_type", "")),
    }


def type_bonus(row: dict[str, str], query_terms: list[str]) -> int:
    bonuses = TYPE_HINTS.get(normalize(row.get("mem_type", "")), {})
    score = 0
    for term in query_terms:
        for hint, weight in bonuses.items():
            if hint in term:
                score += weight
    return score


def score_row(query_terms: list[str], row: dict[str, str], original_tokens: list[str]) -> tuple[int, int, int, int]:
    corpus = row_corpus(row)
    score = 0
    matched_terms = 0
    title_hits = 0
    matched_tokens: set[str] = set()
    for term in query_terms:
        term_tokens = tokenize(term)
        if not term_tokens:
            continue
        term_hit = False
        if term in corpus["title"]:
            score += 70
            title_hits += 1
            term_hit = True
        if term in corpus["summary"]:
            score += 45
            term_hit = True
        if term in corpus["tags"]:
            score += 40
            term_hit = True
        if term in corpus["source"]:
            score += 25
            term_hit = True
        if term == corpus["run"] or term == corpus["stage"]:
            score += 50
            term_hit = True
        for token in term_tokens:
            if token in corpus["title"]:
                score += 16
                title_hits += 1
                term_hit = True
                matched_tokens.add(token)
            if token in corpus["summary"]:
                score += 10
                term_hit = True
                matched_tokens.add(token)
            if token in corpus["tags"]:
                score += 8
                term_hit = True
                matched_tokens.add(token)
            if token in corpus["source"] or token in corpus["run"] or token in corpus["stage"]:
                score += 6
                term_hit = True
                matched_tokens.add(token)
        if term_hit:
            matched_terms += 1
    score += matched_terms * 12
    score += title_hits * 4
    score += type_bonus(row, query_terms)
    if corpus["source_kind"] == "workflow":
        score += 18
    if corpus["type"] == "prompt" and corpus["title"] in GENERIC_PROMPT_TITLES:
        score -= 25
    matched_specific = len([token for token in matched_tokens if token not in GENERIC_TOKENS])
    required_specific = 0
    if len(original_tokens) >= 2:
        required_specific = 1
    if required_specific and matched_specific < required_specific and not any(term in corpus["title"] or term in corpus["summary"] for term in query_terms):
        score = 0
    return score, matched_terms, title_hits, matched_specific


def dedupe_ranked(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            normalize(row.get("mem_type", "")),
            normalize(row.get("title", "")),
            normalize(row.get("summary", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def query_memory(
    *,
    mem_dir: Path,
    query: str,
    mem_type: str = "",
    stage: str = "",
    run_id: str = "",
    limit: int = 10,
) -> list[dict[str, str]]:
    idx_path = mem_dir / "idx.tsv"
    if not idx_path.exists():
        raise SystemExit(f"Missing memory index: {idx_path}")
    rows = load_rows(idx_path)
    query_terms = expand_terms(query)
    original_tokens = tokenize(query)
    filtered: list[tuple[int, int, int, int, dict[str, str]]] = []
    for row in rows:
        if mem_type and normalize(row.get("mem_type", "")) != normalize(mem_type):
            continue
        if stage and normalize(row.get("stage", "")) != normalize(stage):
            continue
        if run_id and normalize(row.get("run_id", "")) != normalize(run_id):
            continue
        score, matched_terms, title_hits, matched_specific = score_row(query_terms, row, original_tokens)
        if query_terms and score == 0:
            continue
        filtered.append((score, matched_terms, title_hits, matched_specific, row))
    filtered.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            -item[2],
            -item[3],
            item[4].get("mem_type", ""),
            item[4].get("ref_id", ""),
            item[4].get("title", ""),
        )
    )
    ranked = [row for _, _, _, _, row in filtered]
    ranked = dedupe_ranked(ranked)
    return ranked[: max(limit, 0)]


def display_text(value: str) -> str:
    text = str(value or "")
    return text.encode("ascii", "replace").decode("ascii")


def render_table(rows: list[dict[str, str]]) -> str:
    cols = ["mem_id", "mem_type", "stage", "run_id", "title", "summary", "source_file"]
    widths = {col: len(col) for col in cols}
    compact_rows: list[dict[str, str]] = []
    for row in rows:
        compact: dict[str, str] = {}
        for col in cols:
            value = display_text(row.get(col, ""))
            if col in {"title", "summary", "source_file"} and len(value) > 56:
                value = value[:53] + "..."
            compact[col] = value
            widths[col] = max(widths[col], len(value))
        compact_rows.append(compact)
    lines = []
    lines.append(" | ".join(col.ljust(widths[col]) for col in cols))
    lines.append("-+-".join("-" * widths[col] for col in cols))
    for row in compact_rows:
        lines.append(" | ".join(row[col].ljust(widths[col]) for col in cols))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    mem_type = TYPE_MAP.get(normalize(args.type))
    if mem_type is None:
        raise SystemExit(f"Unsupported --type: {args.type}")
    result_rows = query_memory(
        mem_dir=args.mem_dir.resolve(),
        query=args.query,
        mem_type=mem_type,
        stage=args.stage,
        run_id=args.run,
        limit=args.limit,
    )
    if args.format == "json":
        print(json.dumps(result_rows, indent=2, ensure_ascii=True))
    else:
        if result_rows:
            print(render_table(result_rows))
        print(f"count={len(result_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
