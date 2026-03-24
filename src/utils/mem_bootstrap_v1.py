#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.utils.paths import DATA_MEM_V1_DIR
    from src.utils.query_mem_v1 import query_memory
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import DATA_MEM_V1_DIR
    from src.utils.query_mem_v1 import query_memory


TASK_MAP = {
    "debugging": {
        "queries": ["collapse", "stage2 parsing"],
        "files": ["AGENTS.md", "project/ACTIVE_PIPELINE_RUNBOOK.md", "src/utils/query_mem_v1.py"],
    },
    "regression": {
        "queries": ["regression", "DOE", "stage2 parsing"],
        "files": ["project/ACTIVE_PIPELINE_FLOW.md", "project/4_DECISIONS_LOG.md", "src/utils/build_mem_v1.py"],
    },
    "run_compare": {
        "queries": ["run lineage", "identity mismatch"],
        "files": ["project/ACTIVE_PIPELINE_RUNBOOK.md", "docs/methods/results_lineage_normalization_pass.md"],
    },
    "pipeline_mod": {
        "queries": ["family variant", "table-first"],
        "files": ["project/2_ARCHITECTURE.md", "project/PIPELINE_SCRIPT_MAP.md", "AGENTS.md"],
    },
    "gt_mismatch": {
        "queries": ["identity mismatch", "BB3JUVW7", "family variant"],
        "files": ["project/ACTIVE_PIPELINE_RUNBOOK.md", "README.md"],
    },
    "lineage": {
        "queries": ["run lineage", "DOE"],
        "files": ["project/ACTIVE_PIPELINE_RUNBOOK.md", "docs/methods/results_lineage_normalization_pass.md"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap governed memory lookup for a complex task.")
    parser.add_argument("--query", required=True, help="Natural-language task description or query.")
    parser.add_argument("--task", choices=sorted(TASK_MAP), default="", help="Optional explicit task class.")
    parser.add_argument("--mem-dir", type=Path, default=DATA_MEM_V1_DIR, help="Memory directory. Default: data/mem/v1")
    parser.add_argument("--limit", type=int, default=5, help="Top memory hits per query.")
    return parser.parse_args()


def detect_task(query: str) -> str:
    lowered = query.lower()
    if "lineage" in lowered or "parent" in lowered or "child" in lowered or "run compare" in lowered:
        return "lineage"
    if "mismatch" in lowered or "gt" in lowered:
        return "gt_mismatch"
    if "regression" in lowered:
        return "regression"
    if "pipeline" in lowered or "modify" in lowered or "change" in lowered:
        return "pipeline_mod"
    if "compare" in lowered or "versus" in lowered:
        return "run_compare"
    return "debugging"


def main() -> int:
    args = parse_args()
    task = args.task or detect_task(args.query)
    config = TASK_MAP[task]
    queries = [args.query, *config["queries"]]
    seen_refs: set[str] = set()
    source_files: list[str] = []
    print(f"task_class={task}")
    for query in queries:
        print(f"query={query}")
        hits = query_memory(mem_dir=args.mem_dir.resolve(), query=query, limit=args.limit)
        for row in hits[: args.limit]:
            print(
                f"hit={row.get('mem_id','')}|{row.get('mem_type','')}|{row.get('title','')}|"
                f"{row.get('source_file','')}|{row.get('summary','')[:120]}"
            )
            source = row.get("source_file", "")
            if source and source not in seen_refs:
                seen_refs.add(source)
                source_files.append(source)
    for path in config["files"]:
        if path not in seen_refs:
            source_files.append(path)
            seen_refs.add(path)
    print("next_files=")
    for path in source_files[:12]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
