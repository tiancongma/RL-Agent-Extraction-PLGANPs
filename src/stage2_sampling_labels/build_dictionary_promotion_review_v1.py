#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GLOBAL_LEXICON = REPO_ROOT / "data" / "cleaned" / "reference" / "value_normalization_lexicon_v1.tsv"

OUTPUT_FIELDS = [
    "field_family",
    "surface_form",
    "canonical_form",
    "supporting_paper_count",
    "supporting_paper_keys",
    "conflicting_canonical_forms",
    "already_global",
    "collision_risk",
    "recommended_action",
    "promotion_basis",
]
ACTIVE_STATUSES = {"approved", "approved_paper_local", "candidate", "current_run_paper_local", "local_only"}
FORMULATION_CODE_RE = re.compile(r"^(?:F\d+|P\d+|NP\d+|BATCH\d*|T\d+|R\d+)$", re.IGNORECASE)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value).lower())


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def is_active_paper_local(row: dict[str, str]) -> bool:
    if normalize_text(row.get("scope")) != "paper_local":
        return False
    if not normalize_text(row.get("paper_key")):
        return False
    status = normalize_text(row.get("status") or row.get("review_status") or row.get("promotion_status")).lower()
    return not status or status in ACTIVE_STATUSES


def build_promotion_review_rows(
    registry_rows: list[dict[str, str]],
    global_lexicon_rows: list[dict[str, str]],
    *,
    min_support_papers: int = 2,
) -> list[dict[str, str]]:
    global_keys = {
        (canonical_key(row.get("field_family", "")), canonical_key(row.get("surface_form", "")), canonical_key(row.get("canonical_form", "")))
        for row in global_lexicon_rows
        if normalize_text(row.get("scope") or "global") == "global"
    }
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"canonical_forms": defaultdict(set), "papers": set()})
    for row in registry_rows:
        if not is_active_paper_local(row):
            continue
        family = normalize_text(row.get("field_family"))
        surface = normalize_text(row.get("surface_form"))
        canonical = normalize_text(row.get("canonical_form"))
        paper_key = normalize_text(row.get("paper_key"))
        if not family or not surface or not canonical or not paper_key:
            continue
        key = (family, surface)
        grouped[key]["canonical_forms"][canonical].add(paper_key)
        grouped[key]["papers"].add(paper_key)

    output: list[dict[str, str]] = []
    for (family, surface), info in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1].lower())):
        canonical_forms: dict[str, set[str]] = info["canonical_forms"]
        papers: set[str] = info["papers"]
        canonical_items = sorted(canonical_forms.items(), key=lambda item: (-len(item[1]), item[0].lower()))
        primary_canonical = canonical_items[0][0]
        conflicts = [canonical for canonical, _paper_keys in canonical_items[1:]]
        already_global = (canonical_key(family), canonical_key(surface), canonical_key(primary_canonical)) in global_keys
        collision_risk = ""
        if conflicts:
            collision_risk = "conflicting_canonical_forms"
        elif FORMULATION_CODE_RE.fullmatch(surface):
            collision_risk = "formulation_code_like_surface"
        elif len(surface) <= 2:
            collision_risk = "short_surface_requires_manual_review"
        support_count = len(papers)
        if already_global:
            action = "already_global"
        elif conflicts:
            action = "keep_paper_local_conflict"
        elif collision_risk:
            action = "manual_review"
        elif support_count >= min_support_papers:
            action = "promote_to_global_candidate"
        else:
            action = "keep_paper_local_insufficient_support"
        output.append(
            {
                "field_family": family,
                "surface_form": surface,
                "canonical_form": primary_canonical,
                "supporting_paper_count": str(support_count),
                "supporting_paper_keys": ";".join(sorted(papers)),
                "conflicting_canonical_forms": ";".join(conflicts),
                "already_global": "yes" if already_global else "no",
                "collision_risk": collision_risk,
                "recommended_action": action,
                "promotion_basis": f"paper_local_support_count={support_count};min_support_papers={min_support_papers}",
            }
        )
    return output


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build review surface for paper-local dictionary promotion candidates.")
    parser.add_argument("--registry", required=True, help="Paper-local abbreviation registry TSV.")
    parser.add_argument("--global-lexicon", default=str(DEFAULT_GLOBAL_LEXICON), help="Current curated global/paper-local lexicon TSV.")
    parser.add_argument("--output", required=True, help="Output promotion review TSV.")
    parser.add_argument("--min-support-papers", type=int, default=2)
    args = parser.parse_args()
    rows = build_promotion_review_rows(
        read_tsv(Path(args.registry)),
        read_tsv(Path(args.global_lexicon)),
        min_support_papers=args.min_support_papers,
    )
    write_tsv(Path(args.output), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
