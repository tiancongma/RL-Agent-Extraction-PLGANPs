#!/usr/bin/env python3
"""
Audit top-level run-directory lineage layout under data/results.

Purpose
- Detect sibling top-level run directories that appear to belong to the same
  experiment lineage because they share the same timestamp and git short hash.
- Accept future MDEC084 top-level v2 bucket roots without treating them as
  malformed legacy runs.
- Emit deterministic audit artifacts that help enforce the repository rule that
  one top-level run directory should represent one benchmark or experiment
  lineage, while retries and repair steps should live under that lineage as
  child executions.

Inputs
- --results-dir: directory containing top-level results entries
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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.utils.paths import DATA_RESULTS_DIR
from src.utils.run_id import classify_results_path


@dataclass(frozen=True)
class RunEntry:
    entry_name: str
    layout_kind: str
    lineage_prefix: str
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


def guess_legacy_role(suffix: str) -> str:
    lowered = str(suffix).lower()
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
        path_info = classify_results_path(path, results_dir=results_dir)
        layout_kind = path_info["path_kind"]
        if layout_kind == "legacy_run_root":
            # Group legacy same-prefix siblings so historical lineage sprawl still audits cleanly.
            parts = path.name.split("_", 4)
            if len(parts) < 5:
                continue
            lineage_prefix = "_".join(parts[:4])
            suffix = parts[4]
            yield RunEntry(
                entry_name=path.name,
                layout_kind=layout_kind,
                lineage_prefix=lineage_prefix,
                role_guess=guess_legacy_role(suffix),
                path=path,
            )
            continue
        if layout_kind == "v2_bucket_root":
            yield RunEntry(
                entry_name=path.name,
                layout_kind=layout_kind,
                lineage_prefix=path.name,
                role_guess="v2_bucket_root",
                path=path,
            )
            continue
        if layout_kind == "v2_child_top_level_invalid":
            yield RunEntry(
                entry_name=path.name,
                layout_kind=layout_kind,
                lineage_prefix=path.name,
                role_guess="misplaced_v2_child_top_level",
                path=path,
            )


def build_groups(entries: Iterable[RunEntry]) -> dict[str, list[RunEntry]]:
    grouped: dict[str, list[RunEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.lineage_prefix].append(entry)
    for lineage_prefix in grouped:
        grouped[lineage_prefix] = sorted(grouped[lineage_prefix], key=lambda e: e.entry_name)
    return dict(sorted(grouped.items()))


def choose_parent_candidate(entries: list[RunEntry]) -> RunEntry:
    non_legacy = [entry for entry in entries if entry.layout_kind != "legacy_run_root"]
    if non_legacy:
        return entries[0]
    benchmark_entries = [entry for entry in entries if "benchmark" in entry.entry_name.lower()]
    if benchmark_entries:
        return benchmark_entries[0]
    complete_entries = [entry for entry in entries if "complete" in entry.entry_name.lower()]
    if complete_entries:
        return complete_entries[0]
    return entries[0]


def write_tsv(out_tsv: Path, groups: dict[str, list[RunEntry]], min_group_size: int) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layout_kind",
                "lineage_prefix",
                "group_size",
                "flagged_as_bloat",
                "entry_name",
                "role_guess",
                "path",
                "recommended_parent_candidate",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for lineage_prefix, entries in groups.items():
            flagged = (
                entries[0].layout_kind == "legacy_run_root"
                and len(entries) >= min_group_size
            )
            parent_candidate = choose_parent_candidate(entries).entry_name
            for entry in entries:
                writer.writerow(
                    {
                        "layout_kind": entry.layout_kind,
                        "lineage_prefix": lineage_prefix,
                        "group_size": len(entries),
                        "flagged_as_bloat": "yes" if flagged else "no",
                        "entry_name": entry.entry_name,
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
    lines.append(
        "This report flags likely legacy top-level lineage sprawl and also classifies future v2 bucket roots conservatively."
    )
    lines.append("")
    for lineage_prefix, entries in groups.items():
        if entries[0].layout_kind == "legacy_run_root" and len(entries) < min_group_size:
            continue
        parent_candidate = choose_parent_candidate(entries)
        lines.append(f"## {lineage_prefix}")
        lines.append("")
        lines.append(f"- layout_kind: `{entries[0].layout_kind}`")
        lines.append(f"- sibling_count: `{len(entries)}`")
        lines.append(f"- recommended_parent_candidate: `{parent_candidate.entry_name}`")
        lines.append("- entries:")
        for entry in entries:
            lines.append(
                f"  - `{entry.entry_name}` ({entry.role_guess}) -> `{entry.path.as_posix()}`"
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
