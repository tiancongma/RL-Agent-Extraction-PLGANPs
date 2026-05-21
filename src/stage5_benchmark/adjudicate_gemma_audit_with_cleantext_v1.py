#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.utils.paths import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import PROJECT_ROOT

csv.field_size_limit(sys.maxsize)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: scalar(row.get(field, "")) for field in fields})


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def norm_exact(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum() or ch in ".+-")


def contains_value(text: str, value: str) -> bool:
    if not text or not value:
        return False
    text_lower = text.lower()
    value_lower = value.lower().strip()
    if value_lower and value_lower in text_lower:
        return True
    needle = norm_exact(value)
    hay = norm_exact(text)
    return bool(needle and needle in hay)


def context_for(text: str, value: str, limit: int = 450) -> str:
    if not text or not value:
        return ""
    lower = text.lower()
    idx = lower.find(value.lower().strip())
    if idx < 0:
        needle = norm_exact(value)
        if not needle:
            return ""
        return ""
    start = max(0, idx - limit // 2)
    end = min(len(text), idx + len(value) + limit // 2)
    return compact_text(text[start:end])


def load_key2txt(path: Path) -> dict[str, Path]:
    rows = read_tsv(path)
    out: dict[str, Path] = {}
    for row in rows:
        key = row.get("key") or row.get("paper_key")
        text_path = row.get("txt_path") or row.get("text_path") or row.get("path")
        if key and text_path:
            out[key] = repo_path(text_path)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Conservatively adjudicate Gemma Evidence Binding audit results against clean text.")
    parser.add_argument("--final-table-tsv", required=True)
    parser.add_argument("--gemma-field-review-tsv", required=True)
    parser.add_argument("--key2txt-tsv", default="data/cleaned/index/key2txt.tsv")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    out_dir = repo_path(args.out_dir)
    analysis = out_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    final_table = repo_path(args.final_table_tsv)
    shutil.copy2(final_table, out_dir / "final_formulation_table_v1.tsv")
    reviews = read_tsv(repo_path(args.gemma_field_review_tsv))
    key2txt = load_key2txt(repo_path(args.key2txt_tsv))
    text_cache: dict[str, str] = {}

    ledger: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    evidence_repairs: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    for row in reviews:
        paper_key = row.get("paper_key", "")
        text_path = key2txt.get(paper_key)
        text = ""
        if text_path:
            if str(text_path) not in text_cache:
                text_cache[str(text_path)] = text_path.read_text(encoding="utf-8", errors="replace")
            text = text_cache[str(text_path)]
        frozen = row.get("frozen_value", "")
        replacement = row.get("replacement_evidence_text", "")
        frozen_found = contains_value(text, frozen)
        replacement_found = contains_value(text, replacement)
        status = row.get("gemma_field_status", "")
        if status == "replacement_evidence_found" and replacement and replacement_found and contains_value(replacement, frozen):
            decision = "repair_evidence_sidecar_only"
            evidence_repairs.append(row)
        elif status == "possible_value_error" and frozen_found:
            decision = "gemma_incorrect_keep_final_value"
        elif status in {"possible_value_error", "unresolved_evidence_defect", "needs_human_review"}:
            decision = "manual_targeted_review_required"
            manual.append(row)
        else:
            decision = "keep_final_value"
        ledger_row = {
            **row,
            "clean_text_path": str(text_path or ""),
            "frozen_value_found_in_clean_text": "yes" if frozen_found else "no",
            "replacement_found_in_clean_text": "yes" if replacement_found else "no",
            "decision": decision,
            "frozen_context": context_for(text, frozen),
            "replacement_context": context_for(text, replacement),
        }
        ledger.append(ledger_row)

    fields = list(ledger[0].keys()) if ledger else []
    write_tsv(analysis / "gemma_cleantext_adjudication_ledger_v1.tsv", ledger, fields)
    write_tsv(analysis / "final_value_correction_ledger_v1.tsv", corrections, fields or ["paper_key", "final_formulation_id"])
    write_tsv(analysis / "evidence_sidecar_repair_ledger_v1.tsv", evidence_repairs, list(evidence_repairs[0].keys()) if evidence_repairs else ["paper_key", "final_formulation_id"])
    write_tsv(analysis / "manual_review_ledger_v1.tsv", manual, list(manual[0].keys()) if manual else ["paper_key", "final_formulation_id"])
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "src/stage5_benchmark/adjudicate_gemma_audit_with_cleantext_v1.py",
        "run_type": "diagnostic clean-text adjudication sidecar",
        "final_table_value_corrections_applied": len(corrections),
        "evidence_sidecar_repairs": len(evidence_repairs),
        "manual_review_rows": len(manual),
        "decision_distribution": dict(Counter(row["decision"] for row in ledger)),
    }
    (analysis / "adjudication_metadata_v1.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "RUN_CONTEXT.md").write_text(
        "\n".join(
            [
                "# RUN_CONTEXT",
                "",
                "## Run purpose",
                "Adjudicate Gemma Evidence Binding audit results against clean text without changing frozen source artifacts.",
                "",
                "## Run type",
                "Diagnostic-only clean-text adjudication sidecar. Not benchmark-valid final output.",
                "",
                "## Boundary",
                "Gemma is treated as an audit signal. This script does not apply value corrections automatically; the emitted final table is an unchanged candidate copy unless a future targeted repair ledger is manually promoted.",
                "",
                "## Inputs",
                f"- final_table_tsv: `{final_table}`",
                f"- gemma_field_review_tsv: `{repo_path(args.gemma_field_review_tsv)}`",
                "",
                "## Counts",
                f"- reviewed_fields: {len(ledger)}",
                f"- final_table_value_corrections_applied: {len(corrections)}",
                f"- evidence_sidecar_repairs: {len(evidence_repairs)}",
                f"- manual_review_rows: {len(manual)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"reviewed_fields={len(ledger)}")
    print(f"final_table_value_corrections_applied={len(corrections)}")
    print(f"manual_review_rows={len(manual)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
