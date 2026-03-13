#!/usr/bin/env python3
"""
Audit top-level run-directory lineage layout under data/results.

Purpose
- Detect sibling top-level run directories that appear to belong to the same
  experiment lineage because they share the same timestamp and git short hash.
- Emit deterministic audit artifacts that help enforce the repository rule that
  one top-level run directory should represent one benchmark or experiment
  lineage, while retries and repair steps should live under that lineage as
  child executions.

Inputs
- --results-dir: directory containing top-level run_* result directories
- --out-tsv: output TSV listing grouped lineages and detected sibling runs
- --out-md: optional Markdown summary of likely lineage sprawl
- --min-group-size: minimum sibling count required before a lineage is flagged

Outputs
- TSV with one row per top-level run directory and lineage grouping metadata
- Optional Markdown report summarizing flagged lineages

Stage role
- This is a cross-cutting governance and audit utility. It is not part of the
  canonical production pipeline stages.

What this script does not do
- It does not move, delete, or rewrite any run directories.
- It does not infer benchmark validity.
- It does not create parent lineage directories automatically.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.utils.paths import DATA_RESULTS_DIR


RUN_ID_PATTERN = re.compile(r"^(run_\d{8}_\d{4}_[0-9a-f]{7})_(.+)$")


@dataclass(frozen=True)
class RunEntry:
    run_name: str
    lineage_prefix: str
    suffix: str
    role_guess: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit top-level run-directory lineage sprawl under data/results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DATA_RESULTS_DIR,
        help="Root directory that contains top-level run_* directories.",
    )
    parser.add_argument(
        "--out-tsv",
        type=Path,
        required=True,
        help="TSV path for the per-run lineage audit table.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional Markdown summary path.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="Minimum top-level sibling count required before a lineage is flagged.",
    )
    return parser.parse_args()


def guess_role(suffix: str) -> str:
    lowered = suffix.lower()
    if "benchmark" in lowered:
        return "benchmark_or_parent"
    if "stage3" in lowered:
        return "stage3_child"
    if "stage2" in lowered and "complete" in lowered:
        return "stage2_complete_child"
    if "stage2" in lowered and "retry" in lowered:
        return "retry_child"
    if "stage2" in lowered and "remaining" in lowered:
        return "partial_recovery_child"
    if "refresh" in lowered:
        return "refresh_child"
    if "analysis" in lowered:
        return "analysis_child"
    if "stage2" in lowered:
        return "stage2_child"
    if "stage5" in lowered:
        return "stage5_child"
    return "unclassified_child"


def iter_run_entries(results_dir: Path) -> Iterable[RunEntry]:
    for path in sorted(results_dir.iterdir(), key=lambda p: p.name):
        if not path.is_dir():
            continue
        match = RUN_ID_PATTERN.match(path.name)
        if not match:
            continue
        lineage_prefix, suffix = match.groups()
        yield RunEntry(
            run_name=path.name,
            lineage_prefix=lineage_prefix,
            suffix=suffix,
            role_guess=guess_role(suffix),
            path=path,
        )


def build_groups(entries: Iterable[RunEntry]) -> dict[str, list[RunEntry]]:
    grouped: dict[str, list[RunEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.lineage_prefix].append(entry)
    for lineage_prefix in grouped:
        grouped[lineage_prefix] = sorted(grouped[lineage_prefix], key=lambda e: e.run_name)
    return dict(sorted(grouped.items()))


def choose_parent_candidate(entries: list[RunEntry]) -> RunEntry:
    benchmark_entries = [entry for entry in entries if "benchmark" in entry.run_name.lower()]
    if benchmark_entries:
        return benchmark_entries[0]
    complete_entries = [entry for entry in entries if "complete" in entry.run_name.lower()]
    if complete_entries:
        return complete_entries[0]
    return entries[0]


def write_tsv(out_tsv: Path, groups: dict[str, list[RunEntry]], min_group_size: int) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lineage_prefix",
                "group_size",
                "flagged_as_bloat",
                "run_name",
                "role_guess",
                "path",
                "recommended_parent_candidate",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for lineage_prefix, entries in groups.items():
            flagged = len(entries) >= min_group_size
            parent_candidate = choose_parent_candidate(entries).run_name
            for entry in entries:
                writer.writerow(
                    {
                        "lineage_prefix": lineage_prefix,
                        "group_size": len(entries),
                        "flagged_as_bloat": "yes" if flagged else "no",
                        "run_name": entry.run_name,
                        "role_guess": entry.role_guess,
                        "path": entry.path.as_posix(),
                        "recommended_parent_candidate": parent_candidate,
                    }
                )


def write_markdown(out_md: Path, groups: dict[str, list[RunEntry]], min_group_size: int) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Run Lineage Layout Audit")
    lines.append("")
    lines.append("This report flags top-level `run_*` sibling groups that likely belong to one lineage.")
    lines.append("")
    for lineage_prefix, entries in groups.items():
        if len(entries) < min_group_size:
            continue
        parent_candidate = choose_parent_candidate(entries)
        lines.append(f"## {lineage_prefix}")
        lines.append("")
        lines.append(f"- sibling_count: `{len(entries)}`")
        lines.append(f"- recommended_parent_candidate: `{parent_candidate.run_name}`")
        lines.append("- sibling_runs:")
        for entry in entries:
            lines.append(
                f"  - `{entry.run_name}` ({entry.role_guess}) -> `{entry.path.as_posix()}`"
            )
        lines.append("")
    if len(lines) == 4:
        lines.append("No top-level lineage groups met the flagging threshold.")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {results_dir}")
    groups = build_groups(iter_run_entries(results_dir))
    write_tsv(args.out_tsv.resolve(), groups, args.min_group_size)
    if args.out_md is not None:
        write_markdown(args.out_md.resolve(), groups, args.min_group_size)


if __name__ == "__main__":
    main()
