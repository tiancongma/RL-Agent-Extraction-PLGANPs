#!/usr/bin/env python3
"""Diagnostic-only selector anchor recall / ranker-not-filter audit.

This helper inspects Stage2 candidate/evidence artifacts and emits a registry that
separates prompt selection from preservation for downstream authority. It does
not run the LLM and does not create benchmark-valid outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def normalize_text(value: Any) -> str:
    """Lightweight local normalizer; avoids importing live Stage2 LLM runtime."""
    return re.sub(r"\s+", " ", str(value or "").strip())


FIELDNAMES = [
    "paper_key",
    "candidate_id",
    "candidate_kind",
    "source_type",
    "origin_locator",
    "selector_rank",
    "priority_score",
    "selected_for_prompt",
    "preserved_for_authority",
    "registry_retained",
    "exclusion_reason",
    "selector_is_authority_filter_violation",
    "text_preview",
]


def _candidate_kind(candidate: dict[str, Any]) -> str:
    explicit = normalize_text(candidate.get("candidate_kind")).lower()
    if explicit:
        return explicit
    ctype = normalize_text(candidate.get("candidate_type")).lower()
    return "table" if ctype == "table" else "paragraph"


def _candidate_id(candidate: dict[str, Any], index: int) -> str:
    return normalize_text(candidate.get("candidate_id")) or f"candidate_{index:04d}"


def _reason_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [normalize_text(item) for item in value if normalize_text(item)]
    return [normalize_text(value)]


def _is_hard_drop(candidate: dict[str, Any], reason: str = "") -> bool:
    inclusion = normalize_text(candidate.get("table_inclusion_class")).lower()
    reason_text = " ".join(_reason_values(candidate.get("hard_drop_reason")) + [normalize_text(reason)]).lower()
    return inclusion == "hard_drop" or "hard_drop" in reason_text or "confirmed_noise" in reason_text


def _explicit_retained(candidate: dict[str, Any], default: bool) -> bool:
    for field in ("registry_retained", "preserved_for_authority"):
        if field not in candidate:
            continue
        value = candidate.get(field)
        if isinstance(value, bool):
            return value
        normalized = normalize_text(value).lower()
        if normalized in {"yes", "true", "1"}:
            return True
        if normalized in {"no", "false", "0"}:
            return False
    return default


def build_selector_recall_rows(
    *,
    paper_key: str,
    candidates: Iterable[dict[str, Any]],
    selected_candidate_ids: set[str],
    suppression_reasons: dict[str, str],
) -> list[dict[str, str]]:
    enriched: list[tuple[float, int, dict[str, Any], str]] = []
    for idx, candidate in enumerate(candidates, start=1):
        cid = _candidate_id(candidate, idx)
        try:
            score = float(candidate.get("priority_score", candidate.get("table_score", candidate.get("score", 0))) or 0)
        except (TypeError, ValueError):
            score = 0.0
        enriched.append((score, idx, candidate, cid))
    enriched.sort(key=lambda item: (-item[0], item[1], item[3]))

    rows: list[dict[str, str]] = []
    for rank, (score, _idx, candidate, cid) in enumerate(enriched, start=1):
        reason = suppression_reasons.get(cid, "")
        selected = cid in selected_candidate_ids
        hard_drop = _is_hard_drop(candidate, reason)
        preserved = _explicit_retained(candidate, default=not hard_drop)
        retained = selected or preserved
        violation = "yes" if (not selected and not preserved and not hard_drop) else "no"
        rows.append(
            {
                "paper_key": paper_key,
                "candidate_id": cid,
                "candidate_kind": _candidate_kind(candidate),
                "source_type": normalize_text(candidate.get("source_type")),
                "origin_locator": normalize_text(candidate.get("origin_locator")),
                "selector_rank": str(rank),
                "priority_score": f"{score:g}",
                "selected_for_prompt": "yes" if selected else "no",
                "preserved_for_authority": "yes" if preserved else "no",
                "registry_retained": "yes" if retained else "no",
                "exclusion_reason": reason if hard_drop or reason else "ranked_but_not_prompt_selected",
                "selector_is_authority_filter_violation": violation,
                "text_preview": normalize_text(candidate.get("text_content"))[:240].replace("\t", " ").replace("\n", " "),
            }
        )
    return rows


def audit_selector_registry(candidate_artifact: dict[str, Any], *, evidence_artifact: dict[str, Any] | None = None) -> list[dict[str, str]]:
    evidence_artifact = evidence_artifact or {}
    paper_key = normalize_text(candidate_artifact.get("paper_key")) or normalize_text(evidence_artifact.get("paper_key"))
    candidates = list(candidate_artifact.get("candidate_blocks") or [])
    if not candidates and candidate_artifact.get("selector_candidates"):
        candidates = list(candidate_artifact.get("selector_candidates") or [])
    selected_ids = {
        normalize_text(block.get("candidate_id"))
        for block in (evidence_artifact.get("evidence_blocks") or [])
        if normalize_text(block.get("candidate_id"))
    }
    selector_debug = evidence_artifact.get("selector_debug") if isinstance(evidence_artifact.get("selector_debug"), dict) else {}
    selected_ids.update(normalize_text(x) for x in (selector_debug.get("selected_candidate_ids") or []) if normalize_text(x))
    suppression_reasons = {
        normalize_text(event.get("candidate_id")): normalize_text(event.get("reason"))
        for event in (selector_debug.get("suppression_events") or [])
        if isinstance(event, dict) and normalize_text(event.get("candidate_id"))
    }
    return build_selector_recall_rows(
        paper_key=paper_key,
        candidates=candidates,
        selected_candidate_ids=selected_ids,
        suppression_reasons=suppression_reasons,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_selector_recall_audit(candidate_root: Path, evidence_root: Path | None, out_dir: Path, repo_root: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for candidate_path in sorted(candidate_root.glob("*/candidate_blocks_v1.json")):
        key = candidate_path.parent.name
        evidence_path = evidence_root / key / "evidence_blocks_v1.json" if evidence_root else None
        evidence = _load_json(evidence_path) if evidence_path and evidence_path.exists() else {}
        rows.extend(audit_selector_registry(_load_json(candidate_path), evidence_artifact=evidence))
    with (out_dir / "selector_anchor_recall_registry_v1.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metrics = {
        "candidate_rows": len(rows),
        "selected_for_prompt": sum(1 for r in rows if r["selected_for_prompt"] == "yes"),
        "preserved_for_authority": sum(1 for r in rows if r["preserved_for_authority"] == "yes"),
        "selector_authority_filter_violations": sum(1 for r in rows if r["selector_is_authority_filter_violation"] == "yes"),
    }
    with (out_dir / "selector_anchor_recall_summary_v1.tsv").open("w", encoding="utf-8") as handle:
        handle.write("metric\tvalue\n")
        for key, value in metrics.items():
            handle.write(f"{key}\t{value}\n")
    generated_at = datetime.now().isoformat(timespec="seconds")
    metadata = {
        "generated_by": "audit_selector_anchor_recall_v1.py",
        "generated_at": generated_at,
        "diagnostic_only": True,
        "benchmark_valid": "no",
        "candidate_root": str(candidate_root.relative_to(repo_root)) if candidate_root.is_absolute() and repo_root in candidate_root.parents else str(candidate_root),
        "evidence_root": str(evidence_root.relative_to(repo_root)) if evidence_root and evidence_root.is_absolute() and repo_root in evidence_root.parents else str(evidence_root or ""),
        "semantics": "selector is audited as ranker/summary producer; non-hard-drop candidates remain registry-retained/preserved for authority",
    }
    (out_dir / "selector_anchor_recall_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (out_dir / "RUN_CONTEXT.md").write_text(
        "# selector_anchor_recall_diagnostic\n\n"
        "- diagnostic_only: yes\n- benchmark_valid: no\n"
        f"- generated_at: {generated_at}\n"
        f"- candidate_root: {metadata['candidate_root']}\n"
        f"- evidence_root: {metadata['evidence_root']}\n"
        "- boundary: Stage2 selector recall/ranker-not-filter audit only; no LLM calls; no Stage5 materialization.\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    write_selector_recall_audit(args.candidate_root, args.evidence_root, args.out_dir, args.repo_root)
    print(f"wrote selector recall audit to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
