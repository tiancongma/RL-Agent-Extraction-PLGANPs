#!/usr/bin/env python3
"""
Audit immediate-child semantics under data/results.

Purpose
- Classify only the immediate children of data/results into governed top-level
  roles so the repository can distinguish canonical parent runs from frozen
  historical exceptions, future v2 bucket roots, review surfaces, archive
  roots, and loose artifacts.
- Emit deterministic audit artifacts for top-level results hygiene.

Inputs
- --results-dir: results root to scan; defaults to data/results from path utils
- --out-tsv: TSV audit output path
- --out-md: Markdown audit output path

Outputs
- results_top_level_audit.tsv
- results_top_level_audit.md

Stage role
- This is a cross-cutting governance utility. It is not part of the scientific
  pipeline and it does not modify runtime pipeline behavior.

What this script does not do
- It does not move, rename, or delete files.
- It does not scan nested descendants.
- It does not infer benchmark validity.
- It does not normalize the results tree.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import DATA_RESULTS_DIR
from src.utils.run_id import classify_results_path, is_valid_legacy_run_id


KNOWN_INDEX_OR_REGISTRY_FILES = {
    "ACTIVE_RUN.json",
    "CURRENT_ENGINEERING_RUNS_INDEX.md",
    "HISTORICAL_NON_COMPLIANT_RUNS_INDEX.md",
    "results_top_level_registry_template.tsv",
    "results_top_level_audit.tsv",
    "results_top_level_audit.md",
}
REVIEW_SURFACE_KEYWORDS = ("review", "audit", "reconciliation")


@dataclass(frozen=True)
class EntryRecord:
    entry_name: str
    entry_path: Path
    entry_kind: str
    is_run_dir: str
    run_id: str
    lineage_prefix: str
    top_level_role: str
    parent_run_id: str
    normalization_action: str
    reason_retained_top_level: str
    governance_note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit immediate-child top-level semantics under data/results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DATA_RESULTS_DIR,
        help="Results root to scan. Only immediate children are inspected.",
    )
    parser.add_argument(
        "--out-tsv",
        type=Path,
        default=DATA_RESULTS_DIR / "results_top_level_audit.tsv",
        help="TSV audit output path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DATA_RESULTS_DIR / "results_top_level_audit.md",
        help="Markdown audit output path.",
    )
    return parser.parse_args()


def load_frozen_exception_names(results_dir: Path) -> set[str]:
    index_path = results_dir / "HISTORICAL_NON_COMPLIANT_RUNS_INDEX.md"
    if not index_path.exists():
        return set()
    text = index_path.read_text(encoding="utf-8")
    return set(re.findall(r"`(run_[^`]+)`", text))


def lineage_prefix_for_name(name: str) -> str:
    if not is_valid_legacy_run_id(name):
        return ""
    parts = name.split("_", 4)
    if len(parts) < 4:
        return ""
    return "_".join(parts[:4])


def choose_parent_candidate(run_names: list[str]) -> str:
    lowered = {name: name.lower() for name in run_names}
    benchmark_names = [name for name in run_names if "benchmark" in lowered[name]]
    if benchmark_names:
        return sorted(benchmark_names)[0]
    complete_names = [name for name in run_names if "complete" in lowered[name]]
    if complete_names:
        return sorted(complete_names)[0]
    return sorted(run_names)[0]


def build_lineage_parent_map(results_dir: Path) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in results_dir.iterdir():
        if not path.is_dir():
            continue
        if not is_valid_legacy_run_id(path.name):
            continue
        prefix = lineage_prefix_for_name(path.name)
        if not prefix:
            continue
        grouped[prefix].append(path.name)
    parent_map: dict[str, str] = {}
    for prefix, run_names in grouped.items():
        parent_map[prefix] = choose_parent_candidate(run_names)
    return parent_map


def classify_entry(
    path: Path,
    results_dir: Path,
    frozen_exception_names: set[str],
    lineage_parent_map: dict[str, str],
) -> EntryRecord:
    name = path.name
    is_dir = path.is_dir()
    is_file = path.is_file()
    path_info = classify_results_path(path, results_dir=results_dir)
    path_kind = path_info["path_kind"]
    is_run_dir = "yes" if is_dir and path_kind in {"legacy_run_root", "v2_bucket_root"} else "no"
    run_id = name if path_kind == "legacy_run_root" else ""
    lineage_prefix = lineage_prefix_for_name(name) if run_id else ""

    if is_dir and name == "historical_non_compliant_runs":
        return EntryRecord(
            entry_name=name,
            entry_path=path,
            entry_kind="directory",
            is_run_dir=is_run_dir,
            run_id=run_id,
            lineage_prefix=lineage_prefix,
            top_level_role="archive_root",
            parent_run_id="",
            normalization_action="keep_top_level",
            reason_retained_top_level="governed historical archive root",
            governance_note="Contains separated historical non-compliant runs.",
        )

    if is_dir and name in frozen_exception_names:
        return EntryRecord(
            entry_name=name,
            entry_path=path,
            entry_kind="directory",
            is_run_dir=is_run_dir,
            run_id=run_id,
            lineage_prefix=lineage_prefix,
            top_level_role="frozen_historical_exception",
            parent_run_id="",
            normalization_action="retain_until_reference_update",
            reason_retained_top_level="documented historical compatibility exception",
            governance_note="Named in HISTORICAL_NON_COMPLIANT_RUNS_INDEX.md as retained top-level.",
        )

    if path_kind == "legacy_run_root":
        parent_candidate = lineage_parent_map.get(lineage_prefix, run_id)
        if parent_candidate != run_id:
            return EntryRecord(
                entry_name=name,
                entry_path=path,
                entry_kind="directory",
                is_run_dir=is_run_dir,
                run_id=run_id,
                lineage_prefix=lineage_prefix,
                top_level_role="lineage_child_run",
                parent_run_id=parent_candidate,
                normalization_action="nest_under_parent",
                reason_retained_top_level="same-lineage sibling group detected",
                governance_note="Top-level child run should move under the chosen parent lineage.",
            )
        return EntryRecord(
            entry_name=name,
            entry_path=path,
            entry_kind="directory",
            is_run_dir=is_run_dir,
            run_id=run_id,
            lineage_prefix=lineage_prefix,
            top_level_role="canonical_parent_run",
            parent_run_id="",
            normalization_action="keep_top_level",
            reason_retained_top_level="independent or chosen parent run lineage entry point",
            governance_note="Top-level run parent is allowed when it is the authoritative lineage entry point.",
        )

    if is_dir and path_kind == "v2_bucket_root":
        return EntryRecord(
            entry_name=name,
            entry_path=path,
            entry_kind="directory",
            is_run_dir=is_run_dir,
            run_id="",
            lineage_prefix="",
            top_level_role="future_v2_bucket_root",
            parent_run_id="",
            normalization_action="keep_top_level",
            reason_retained_top_level="future governed bucket root",
            governance_note="Accepted MDEC084 future bucket root under data/results/.",
        )

    if is_dir and path_kind == "v2_child_top_level_invalid":
        return EntryRecord(
            entry_name=name,
            entry_path=path,
            entry_kind="directory",
            is_run_dir=is_run_dir,
            run_id="",
            lineage_prefix="",
            top_level_role="misplaced_v2_child_execution",
            parent_run_id="",
            normalization_action="review_required",
            reason_retained_top_level="child execution folder pattern found at top level",
            governance_note="MDEC084 child execution folders are valid only inside a v2 bucket root.",
        )

    if is_dir and any(keyword in name.lower() for keyword in REVIEW_SURFACE_KEYWORDS):
        return EntryRecord(
            entry_name=name,
            entry_path=path,
            entry_kind="directory",
            is_run_dir=is_run_dir,
            run_id="",
            lineage_prefix="",
            top_level_role="non_run_review_surface",
            parent_run_id="",
            normalization_action="review_required",
            reason_retained_top_level="human-facing review or audit surface",
            governance_note="Allowed during audit; future normalization may place it under a governed non-run namespace.",
        )

    if is_file and name in KNOWN_INDEX_OR_REGISTRY_FILES:
        return EntryRecord(
            entry_name=name,
            entry_path=path,
            entry_kind="file",
            is_run_dir=is_run_dir,
            run_id="",
            lineage_prefix="",
            top_level_role="index_or_registry_file",
            parent_run_id="",
            normalization_action="keep_top_level",
            reason_retained_top_level="governs or indexes the results root",
            governance_note="Top-level governance surface for data/results discovery or audit.",
        )

    if is_file:
        return EntryRecord(
            entry_name=name,
            entry_path=path,
            entry_kind="file",
            is_run_dir=is_run_dir,
            run_id="",
            lineage_prefix="",
            top_level_role="non_run_artifact",
            parent_run_id="",
            normalization_action="review_required",
            reason_retained_top_level="loose top-level results artifact",
            governance_note="Likely normalization candidate unless a governed exception is documented.",
        )

    if is_dir:
        return EntryRecord(
            entry_name=name,
            entry_path=path,
            entry_kind="directory",
            is_run_dir=is_run_dir,
            run_id="",
            lineage_prefix="",
            top_level_role="review_required",
            parent_run_id="",
            normalization_action="review_required",
            reason_retained_top_level="top-level directory not resolved by current conservative rules",
            governance_note="Manual review needed before any move or retention decision.",
        )

    return EntryRecord(
        entry_name=name,
        entry_path=path,
        entry_kind="other",
        is_run_dir=is_run_dir,
        run_id="",
        lineage_prefix="",
        top_level_role="review_required",
        parent_run_id="",
        normalization_action="review_required",
        reason_retained_top_level="unclassified filesystem entry",
        governance_note="Manual review needed.",
    )


def write_tsv(out_path: Path, records: list[EntryRecord]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "entry_name",
                "entry_path",
                "entry_kind",
                "is_run_dir",
                "run_id",
                "lineage_prefix",
                "top_level_role",
                "parent_run_id",
                "normalization_action",
                "reason_retained_top_level",
                "governance_note",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "entry_name": record.entry_name,
                    "entry_path": record.entry_path.as_posix(),
                    "entry_kind": record.entry_kind,
                    "is_run_dir": record.is_run_dir,
                    "run_id": record.run_id,
                    "lineage_prefix": record.lineage_prefix,
                    "top_level_role": record.top_level_role,
                    "parent_run_id": record.parent_run_id,
                    "normalization_action": record.normalization_action,
                    "reason_retained_top_level": record.reason_retained_top_level,
                    "governance_note": record.governance_note,
                }
            )


def write_markdown(out_path: Path, records: list[EntryRecord]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    role_counts = Counter(record.top_level_role for record in records)
    lines: list[str] = [
        "# Results Top-Level Audit",
        "",
        "This report classifies only the immediate children of `data/results/`.",
        "",
        "## Role Counts",
        "",
    ]
    for role, count in sorted(role_counts.items()):
        lines.append(f"- `{role}`: `{count}`")
    lines.extend(
        [
            "",
            "## Entries",
            "",
            "| entry_name | entry_kind | top_level_role | normalization_action | parent_run_id | note |",
            "|---|---|---|---|---|---|",
        ]
    )
    for record in records:
        lines.append(
            "| {name} | {kind} | {role} | {action} | {parent} | {note} |".format(
                name=record.entry_name,
                kind=record.entry_kind,
                role=record.top_level_role,
                action=record.normalization_action,
                parent=record.parent_run_id or "",
                note=record.governance_note,
            )
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {results_dir}")

    frozen_exception_names = load_frozen_exception_names(results_dir)
    lineage_parent_map = build_lineage_parent_map(results_dir)

    records = [
        classify_entry(path, results_dir, frozen_exception_names, lineage_parent_map)
        for path in sorted(results_dir.iterdir(), key=lambda child: child.name.lower())
    ]
    write_tsv(args.out_tsv.resolve(), records)
    write_markdown(args.out_md.resolve(), records)


if __name__ == "__main__":
    main()
