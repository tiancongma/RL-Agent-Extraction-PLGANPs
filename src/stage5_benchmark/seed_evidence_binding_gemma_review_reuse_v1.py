#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.stage5_benchmark.run_evidence_binding_gemma_review_v1 import (
        CORE_VALUE_FIELDS,
        FIELD_FIELDS,
        SUMMARY_FIELDS,
        summarize_field_reviews,
    )
    from src.utils.paths import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage5_benchmark.run_evidence_binding_gemma_review_v1 import (
        CORE_VALUE_FIELDS,
        FIELD_FIELDS,
        SUMMARY_FIELDS,
        summarize_field_reviews,
    )
    from src.utils.paths import PROJECT_ROOT

csv.field_size_limit(sys.maxsize)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: scalar(row.get(field, "")) for field in fields})


def pack_signature(pack: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(pack.get("paper_key", "")).strip(),
        str(pack.get("field_name", "")).strip(),
        str(pack.get("frozen_value", "")).strip(),
        str(pack.get("value_evidence_text", "")).strip(),
        str(pack.get("row_identity_evidence_text", "")).strip(),
        str(pack.get("source_cell_text", "")).strip(),
    )


def risk_fields_by_fid(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("risk_level") not in {"high", "medium"}:
            continue
        if row.get("frozen_value_present") != "yes":
            continue
        if row.get("field_name") not in CORE_VALUE_FIELDS:
            continue
        grouped[row.get("final_formulation_id", "")].append(row)
    return grouped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed exact Evidence Binding Gemma review reuse rows before live Gemma calls.")
    parser.add_argument("--current-packs-jsonl", required=True)
    parser.add_argument("--current-field-risk-tsv", required=True)
    parser.add_argument("--current-row-review-queue-tsv", required=True)
    parser.add_argument("--previous-packs-jsonl", required=True)
    parser.add_argument("--previous-field-review-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    out_dir = repo_path(args.out_dir)
    analysis = out_dir / "analysis"
    current_packs = read_jsonl(repo_path(args.current_packs_jsonl))
    previous_packs = read_jsonl(repo_path(args.previous_packs_jsonl))
    previous_reviews = read_tsv(repo_path(args.previous_field_review_tsv))
    current_field_risk = read_tsv(repo_path(args.current_field_risk_tsv))
    current_row_queue = read_tsv(repo_path(args.current_row_review_queue_tsv))

    current_pack_by_fid_field = {(p.get("final_formulation_id", ""), p.get("field_name", "")): p for p in current_packs}
    prev_sig_by_fid_field = {(p.get("final_formulation_id", ""), p.get("field_name", "")): pack_signature(p) for p in previous_packs}
    prev_review_by_sig: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {}
    for review in previous_reviews:
        sig = prev_sig_by_fid_field.get((review.get("final_formulation_id", ""), review.get("field_name", "")))
        if sig:
            prev_review_by_sig[sig] = review

    risk_by_fid = risk_fields_by_fid(current_field_risk)
    queue_by_fid = {row.get("final_formulation_id", ""): row for row in current_row_queue}
    seeded_summary: list[dict[str, Any]] = []
    seeded_fields: list[dict[str, Any]] = []
    reused_fids: set[str] = set()

    for fid, target_fields in sorted(risk_by_fid.items()):
        reused_reviews: list[dict[str, Any]] = []
        for target in target_fields:
            pack = current_pack_by_fid_field.get((fid, target.get("field_name", "")))
            review = prev_review_by_sig.get(pack_signature(pack or {}))
            if not review:
                reused_reviews = []
                break
            reused_reviews.append(
                {
                    "field_name": target.get("field_name", ""),
                    "field_status": review.get("gemma_field_status", ""),
                    "replacement_evidence_text": review.get("replacement_evidence_text", ""),
                    "replacement_evidence_source": review.get("replacement_evidence_source", ""),
                    "reason": review.get("reason", ""),
                    "recommended_next_action": review.get("recommended_next_action", ""),
                    "previous_raw_response_path": review.get("raw_response_path", ""),
                }
            )
        if not reused_reviews:
            continue
        row = queue_by_fid.get(fid, {})
        counts = summarize_field_reviews(reused_reviews)
        possible = counts.get("possible_value_error_count", 0)
        unresolved = counts.get("unresolved_evidence_defect_count", 0) + counts.get("needs_human_review_count", 0)
        if possible:
            verdict = "possible_value_error"
        elif unresolved:
            verdict = "some_fields_unresolved"
        else:
            verdict = "all_high_risk_fields_resolved"
        seeded_summary.append(
            {
                "paper_key": row.get("paper_key", ""),
                "final_formulation_id": fid,
                "row_risk_level": row.get("row_risk_level", ""),
                "review_priority": row.get("review_priority", ""),
                "audit_model": "reused_exact_pack_signature",
                "audit_status": "reused",
                "formulation_verdict": verdict,
                **counts,
                "recommended_next_action": "reuse_previous_review",
                "reason": "All target field pack signatures exactly matched a previous Gemma review.",
                "prompt_path": "",
                "raw_response_path": ";".join(sorted({r.get("previous_raw_response_path", "") for r in reused_reviews if r.get("previous_raw_response_path")})),
            }
        )
        for review in reused_reviews:
            target = next((item for item in target_fields if item.get("field_name") == review.get("field_name")), {})
            pack = current_pack_by_fid_field.get((fid, review.get("field_name", "")), {})
            seeded_fields.append(
                {
                    "paper_key": row.get("paper_key", ""),
                    "final_formulation_id": fid,
                    "field_name": review.get("field_name", ""),
                    "frozen_value": pack.get("frozen_value", ""),
                    "risk_level": target.get("risk_level", ""),
                    "evidence_status": target.get("evidence_status", ""),
                    "gemma_field_status": review.get("field_status", ""),
                    "replacement_evidence_text": review.get("replacement_evidence_text", ""),
                    "replacement_evidence_source": review.get("replacement_evidence_source", ""),
                    "reason": review.get("reason", ""),
                    "recommended_next_action": review.get("recommended_next_action", ""),
                    "audit_model": "reused_exact_pack_signature",
                    "audit_status": "reused",
                    "raw_response_path": review.get("previous_raw_response_path", ""),
                }
            )
        reused_fids.add(fid)

    write_tsv(analysis / "gemma_evidence_binding_formulation_reviews_v1.tsv", seeded_summary, SUMMARY_FIELDS)
    write_tsv(analysis / "gemma_evidence_binding_field_reviews_v1.tsv", seeded_fields, FIELD_FIELDS)
    remaining_queue = [row for row in current_row_queue if row.get("final_formulation_id", "") not in reused_fids]
    write_tsv(analysis / "gemma_reuse_remaining_row_review_queue_v1.tsv", remaining_queue, list(current_row_queue[0].keys()) if current_row_queue else [])
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "src/stage5_benchmark/seed_evidence_binding_gemma_review_reuse_v1.py",
        "reused_formulation_count": len(seeded_summary),
        "reused_field_count": len(seeded_fields),
        "remaining_row_count": len(remaining_queue),
        "verdict_distribution": dict(Counter(row["formulation_verdict"] for row in seeded_summary)),
    }
    (analysis / "gemma_reuse_seed_metadata_v1.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"reused_formulation_count={meta['reused_formulation_count']}")
    print(f"reused_field_count={meta['reused_field_count']}")
    print(f"remaining_row_count={meta['remaining_row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
