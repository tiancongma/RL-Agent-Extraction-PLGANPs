#!/usr/bin/env python3
"""Audit generic preparation evidence sufficiency and first failure boundary.

Diagnostic-only: this script reads governed cleaned-text-derived Stage2 artifacts
(candidate/evidence blocks) and does not search raw/source files or protocol excerpts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

PREPARATION_ACTION_RE = re.compile(
    r"\b(prepar(?:ed|ation)|dissolv(?:ed|ing)|weigh(?:ed|ing)|mix(?:ed|ing)|poured|added|dropwise|emulsif(?:ied|ication)|stirr(?:ed|ing)|evaporat(?:ed|ion)|nanoprecipitation|solvent\s+(?:diffusion|displacement|evaporation)|organic\s+phase|aqueous\s+phase|form(?:ed|ulation))\b",
    re.I,
)
VALUE_UNIT_RE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*(?:mg|µg|ug|g|ml|mL|µL|uL|%|rpm|°C|h|min)\b|\([^)]*\b\d+(?:\.\d+)?\s*(?:mg|µg|ug|g|ml|mL|%)[^)]*\))",
    re.I,
)
MATERIAL_RE = re.compile(
    r"\b(polymer|plga|pla|pcl|drug|surfactant|stabilizer|stabiliser|poloxamer|pva|tpgs|polysorbate|organic\s+solution|aqueous\s+solution)\b",
    re.I,
)
METHOD_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s*)?(?:materials\s+and\s+methods|methods?|preparation|preparations?|production|fabrication|formulation|synthesis)\b",
    re.I,
)
SECTION_HEADING_RE = re.compile(r"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s*)?[A-Z][A-Za-z0-9 /,()\-]{2,80}\s*(?:\n|$)")

TABLE_NOISE_RE = re.compile(
    r"\b(AUC|Cmax|Tmax|MRT|t1/?2|\bTe\b|pharmacokinetic|pharmacokinetics|targeting parameters|tissue|organ distribution|intravenous administration|release profile|in-?vitro release|cell uptake|cytotoxicity|references|bibliography)\b",
    re.I,
)
FORMULATION_ROLE_RE = re.compile(r"\b(formulation table|formulation[-_ ]?composition|formulation|composition table)\b", re.I)
COMPOSITION_CUE_RE = re.compile(
    r"\b(formulation\s*(?:id|no|number)|polymer\s*(?:mass|amount|mg)|drug\s*(?:mass|amount|mg)|surfactant|stabilizer|stabiliser|organic\s+phase|aqueous\s+phase|phase\s+volume|preparation\s+variable)\b",
    re.I,
)

FIELDNAMES = [
    "paper_key",
    "cleaned_text_path",
    "cleaned_text_has_method_heading",
    "cleaned_text_has_method_body_after_heading",
    "cleaned_text_has_preparation_core",
    "candidate_has_preparation_core",
    "selected_has_preparation_core",
    "preparation_core_candidate_block_ids",
    "preparation_core_selected_block_ids",
    "selected_method_block_ids",
    "table_blocks_selected_count",
    "table_noise_selected_count",
    "pharmacokinetic_table_selected_as_formulation_count",
    "release_or_tissue_table_selected_as_formulation_count",
    "source_quality_status",
    "evidence_selection_status",
    "first_failure_boundary",
    "abstraction_compliance_status",
    "notes",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _text_of(block: dict[str, Any]) -> str:
    return str(block.get("text_content") or block.get("rendered_text") or block.get("text_preview") or "")


def _preparation_core_windows(text: str) -> list[str]:
    """Return local prose windows used for preparation-core co-locality checks.

    The source-quality audit must not let one navigation/caption/reference token
    satisfy action+material+unit across the whole cleaned document.  A true
    preparation core needs those signals in the same local prose block.
    """
    normalized = str(text or "").replace("\r", "\n")
    raw_blocks = [block.strip() for block in re.split(r"\n\s*\n+|\n", normalized) if block.strip()]
    windows: list[str] = []
    for block in raw_blocks:
        compact = re.sub(r"\s+", " ", block).strip()
        if not compact:
            continue
        lower = compact.lower()
        if re.match(r"^(?:table|fig\.?|figure|download|references|cited by|outline|abstract|keywords)\b", lower):
            continue
        if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z /,()\-]{2,80}$", compact):
            continue
        if len(compact) <= 90 and not VALUE_UNIT_RE.search(compact):
            continue
        if len(compact) <= 1800:
            windows.append(compact)
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", compact) if s.strip()]
        for idx in range(len(sentences)):
            window = " ".join(sentences[idx : idx + 3]).strip()
            if window:
                windows.append(window)
    return windows


def has_preparation_core(text: str) -> bool:
    """Generic preparation-core detector: co-local action/context + material/entity + value/unit."""
    for window in _preparation_core_windows(text):
        if PREPARATION_ACTION_RE.search(window) and MATERIAL_RE.search(window) and VALUE_UNIT_RE.search(window):
            return True
    return False


def has_method_heading(text: str) -> bool:
    return bool(METHOD_HEADING_RE.search(text))


def has_method_body_after_heading(text: str) -> bool:
    match = METHOD_HEADING_RE.search(text)
    if not match:
        return False
    tail = text[match.end() : match.end() + 1600]
    # Remove immediate heading-like fragments and require substantive prep/value content.
    return has_preparation_core(tail)


def is_table_block(block: dict[str, Any]) -> bool:
    return str(block.get("block_type") or block.get("evidence_kind") or block.get("source_type") or "").lower().find("table") >= 0


def is_table_noise(text: str) -> bool:
    return bool(TABLE_NOISE_RE.search(text))


def is_misclassified_formulation_table(text: str) -> bool:
    return bool(TABLE_NOISE_RE.search(text) and FORMULATION_ROLE_RE.search(text) and not COMPOSITION_CUE_RE.search(text))


def _resolve_cleaned_text_path(candidate_payload: dict[str, Any], evidence_payload: dict[str, Any], repo_root: Path) -> tuple[str, str]:
    raw = candidate_payload.get("source_clean_text_path") or evidence_payload.get("source_clean_text_path") or ""
    if not raw:
        return "", ""
    path = Path(str(raw))
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        return str(raw), ""
    return str(raw), path.read_text(encoding="utf-8", errors="replace")


def audit_one_paper(*, candidate_path: Path, evidence_path: Path, repo_root: Path | None = None) -> dict[str, str]:
    repo_root = repo_root or Path.cwd()
    candidate_payload = _load_json(candidate_path)
    evidence_payload = _load_json(evidence_path)
    paper_key = str(candidate_payload.get("paper_key") or evidence_payload.get("paper_key") or candidate_path.parent.name)
    clean_path, clean_text = _resolve_cleaned_text_path(candidate_payload, evidence_payload, repo_root)
    candidates = candidate_payload.get("candidate_blocks") or []
    evidence_blocks = evidence_payload.get("evidence_blocks") or []

    prep_candidate_ids = [str(b.get("candidate_id") or b.get("block_id") or "") for b in candidates if has_preparation_core(_text_of(b))]
    prep_selected_ids = [str(b.get("block_id") or b.get("candidate_id") or "") for b in evidence_blocks if has_preparation_core(_text_of(b))]
    selected_method_ids = [str(b.get("block_id") or b.get("candidate_id") or "") for b in evidence_blocks if "method" in str(b.get("block_type") or b.get("evidence_kind") or b.get("block_id") or "").lower()]

    table_blocks = [b for b in evidence_blocks if is_table_block(b)]
    table_noise = [b for b in table_blocks if is_table_noise(_text_of(b))]
    misclassified = [b for b in table_blocks if is_misclassified_formulation_table(_text_of(b))]
    release_or_tissue = [b for b in table_noise if re.search(r"\b(release|tissue|organ distribution|cell uptake)\b", _text_of(b), re.I)]

    clean_has_heading = has_method_heading(clean_text)
    clean_has_body = has_method_body_after_heading(clean_text)
    clean_has_core = has_preparation_core(clean_text)
    candidate_has_core = bool(prep_candidate_ids)
    selected_has_core = bool(prep_selected_ids)

    source_quality_status = "pass"
    evidence_selection_status = "pass"
    first_failure = "pass"
    notes: list[str] = []

    if clean_has_heading and not clean_has_body:
        source_quality_status = "cleaned_text_missing_method_body"
        first_failure = "cleaned_text_missing_method_body"
    elif not clean_has_core and (candidate_has_core or selected_has_core):
        # Should be rare; artifacts may be synthetic or text path unresolved.
        source_quality_status = "cleaned_text_unresolved_or_inconsistent"
        notes.append("candidate/evidence core present but cleaned text path missing or inconsistent")
    elif not clean_has_core:
        source_quality_status = "cleaned_text_missing_preparation_core"
        first_failure = "cleaned_text_missing_preparation_core"
    elif clean_has_core and not candidate_has_core:
        evidence_selection_status = "candidate_segmentation_missing_preparation_core"
        first_failure = "candidate_segmentation_missing_preparation_core"
    elif candidate_has_core and not selected_has_core:
        evidence_selection_status = "evidence_selection_missing_preparation_core"
        first_failure = "evidence_selection_missing_preparation_core"

    if misclassified:
        evidence_selection_status = "table_role_misclassified"
        if first_failure == "pass" or first_failure.startswith("evidence_selection"):
            first_failure = "table_role_misclassified"
    elif table_noise and first_failure == "pass":
        evidence_selection_status = "table_selector_noise_overselected"
        first_failure = "table_selector_noise_overselected"
    elif first_failure == "pass" and selected_has_core:
        first_failure = "materialization_or_carrythrough_boundary"

    return {
        "paper_key": paper_key,
        "cleaned_text_path": clean_path,
        "cleaned_text_has_method_heading": "yes" if clean_has_heading else "no",
        "cleaned_text_has_method_body_after_heading": "yes" if clean_has_body else "no",
        "cleaned_text_has_preparation_core": "yes" if clean_has_core else "no",
        "candidate_has_preparation_core": "yes" if candidate_has_core else "no",
        "selected_has_preparation_core": "yes" if selected_has_core else "no",
        "preparation_core_candidate_block_ids": ";".join(x for x in prep_candidate_ids if x),
        "preparation_core_selected_block_ids": ";".join(x for x in prep_selected_ids if x),
        "selected_method_block_ids": ";".join(x for x in selected_method_ids if x),
        "table_blocks_selected_count": str(len(table_blocks)),
        "table_noise_selected_count": str(len(table_noise)),
        "pharmacokinetic_table_selected_as_formulation_count": str(len(misclassified)),
        "release_or_tissue_table_selected_as_formulation_count": str(len(release_or_tissue)),
        "source_quality_status": source_quality_status,
        "evidence_selection_status": evidence_selection_status,
        "first_failure_boundary": first_failure,
        "abstraction_compliance_status": "generic_only",
        "notes": "; ".join(notes),
    }


def _find_pairs(root: Path, keys: list[str]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    cand_root = root / "semantic_stage2_objects" / "candidate_blocks"
    ev_root = root / "semantic_stage2_objects" / "evidence_blocks"
    if keys:
        iter_keys = keys
    elif cand_root.exists():
        iter_keys = sorted(p.name for p in cand_root.iterdir() if p.is_dir())
    else:
        iter_keys = []
    for key in iter_keys:
        c = cand_root / key / "candidate_blocks_v1.json"
        e = ev_root / key / "evidence_blocks_v1.json"
        if c.exists() and e.exists():
            pairs.append((c, e))
    return pairs


def write_outputs(rows: list[dict[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "preparation_evidence_sufficiency_audit_v1.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, Any] = {"row_count": len(rows), "first_failure_boundary_counts": {}}
    for row in rows:
        key = row["first_failure_boundary"]
        summary["first_failure_boundary_counts"][key] = summary["first_failure_boundary_counts"].get(key, 0) + 1
    (out_dir / "preparation_evidence_sufficiency_summary_v1.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "RUN_CONTEXT.md").write_text(
        "# Preparation Evidence Sufficiency Audit\n\n"
        "benchmark_valid=no\n\n"
        "Diagnostic-only audit of generic cleaned-text/evidence/materialization boundaries. "
        "This audit consumes cleaned-text-derived Stage2 candidate/evidence artifacts only and does not search raw source files or protocol excerpts.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-run-dir", type=Path, help="Stage2 run directory containing semantic_stage2_objects/.")
    parser.add_argument("--candidate-path", type=Path, help="Single candidate_blocks_v1.json path.")
    parser.add_argument("--evidence-path", type=Path, help="Single evidence_blocks_v1.json path.")
    parser.add_argument("--paper-key", action="append", default=[], help="Paper key to audit when --stage2-run-dir is used. May repeat.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    pairs: list[tuple[Path, Path]]
    if args.candidate_path or args.evidence_path:
        if not args.candidate_path or not args.evidence_path:
            raise SystemExit("--candidate-path and --evidence-path must be provided together")
        pairs = [(args.candidate_path, args.evidence_path)]
    elif args.stage2_run_dir:
        pairs = _find_pairs(args.stage2_run_dir, args.paper_key)
    else:
        raise SystemExit("provide --stage2-run-dir or --candidate-path/--evidence-path")
    rows = [audit_one_paper(candidate_path=c, evidence_path=e, repo_root=args.repo_root) for c, e in pairs]
    write_outputs(rows, args.out_dir)
    print(json.dumps({"rows": len(rows), "out_dir": str(args.out_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
