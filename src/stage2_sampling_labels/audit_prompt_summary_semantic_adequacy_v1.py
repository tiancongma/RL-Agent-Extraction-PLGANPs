#!/usr/bin/env python3
"""Diagnostic-only prompt summary semantic adequacy audit.

Prompt summary adequacy is semantic: it checks whether selected Stage2 evidence
surfaces identity/process/formulation cues for LLM semantic discovery. It does
not require complete numeric rows and does not confer value authority.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def normalize_text(value: Any) -> str:
    """Lightweight local normalizer; avoids importing live Stage2 LLM runtime."""
    return re.sub(r"\s+", " ", str(value or "").strip())

IDENTITY_CUES = ("plga", "polymer", "nanoparticle", "nanosphere", "nanocapsule", "pcl", "pla")
PROCESS_CUES = ("prepared", "preparation", "fabricated", "solvent", "emulsion", "evaporation", "nanoprecipitation")
FORMULATION_CUES = ("formulation", "composition", "surfactant", "pva", "drug", "loading", "ratio", "concentration", "table")

FIELDNAMES = [
    "paper_key",
    "block_id",
    "candidate_id",
    "block_type",
    "semantic_adequacy",
    "first_failure_reason",
    "has_identity_signal",
    "has_process_signal",
    "has_formulation_signal",
    "numeric_token_count",
    "requires_full_numeric_rows",
    "summary_view_is_lossy",
    "benchmark_valid",
    "value_authority",
    "text_preview",
]


def _has_any(text: str, cues: tuple[str, ...]) -> bool:
    lower = text.lower()
    for cue in cues:
        if len(cue) <= 4 and cue.isalpha():
            if re.search(rf"(?<![a-z0-9]){re.escape(cue.lower())}(?![a-z0-9])", lower):
                return True
        elif cue in lower:
            return True
    return False


def assess_prompt_summary_semantic_adequacy(block: dict[str, Any]) -> dict[str, str]:
    text = normalize_text(block.get("text_content"))
    has_identity = _has_any(text, IDENTITY_CUES)
    has_process = _has_any(text, PROCESS_CUES)
    has_formulation = _has_any(text, FORMULATION_CUES) or bool(re.search(r"\bf\d{1,3}\b", text, flags=re.I))
    numeric_count = len(re.findall(r"[-+]?\d+(?:\.\d+)?", text))
    block_type = normalize_text(block.get("block_type")) or normalize_text(block.get("evidence_kind"))
    adequate = False
    failure = ""
    if not text:
        failure = "empty_prompt_surface"
    elif block_type == "metadata":
        adequate = True
    elif has_identity and (has_process or has_formulation):
        adequate = True
    elif has_process and has_formulation:
        adequate = True
    else:
        failure = "missing_identity_or_process_signal"
    return {
        "semantic_adequacy": "adequate" if adequate else "inadequate",
        "first_failure_reason": failure,
        "has_identity_signal": "yes" if has_identity else "no",
        "has_process_signal": "yes" if has_process else "no",
        "has_formulation_signal": "yes" if has_formulation else "no",
        "numeric_token_count": str(numeric_count),
        "requires_full_numeric_rows": "no",
        "summary_view_is_lossy": "yes" if bool(block.get("summary_is_lossy")) else "no",
        "benchmark_valid": "no",
        "value_authority": "no",
    }


def write_prompt_summary_audit(evidence_root: Path, out_dir: Path, repo_root: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for evidence_path in sorted(evidence_root.glob("*/evidence_blocks_v1.json")):
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        paper_key = normalize_text(payload.get("paper_key")) or evidence_path.parent.name
        for block in payload.get("evidence_blocks") or []:
            assessment = assess_prompt_summary_semantic_adequacy(block)
            rows.append(
                {
                    "paper_key": paper_key,
                    "block_id": normalize_text(block.get("block_id")),
                    "candidate_id": normalize_text(block.get("candidate_id")),
                    "block_type": normalize_text(block.get("block_type")),
                    **assessment,
                    "text_preview": normalize_text(block.get("text_content"))[:240].replace("\t", " ").replace("\n", " "),
                }
            )
    with (out_dir / "prompt_summary_semantic_adequacy_v1.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    first_failures: dict[str, int] = {}
    for row in rows:
        if row["semantic_adequacy"] == "inadequate":
            first_failures[row["first_failure_reason"]] = first_failures.get(row["first_failure_reason"], 0) + 1
    with (out_dir / "prompt_summary_semantic_adequacy_summary_v1.tsv").open("w", encoding="utf-8") as handle:
        handle.write("metric\tvalue\n")
        handle.write(f"evidence_blocks\t{len(rows)}\n")
        handle.write(f"adequate\t{sum(1 for r in rows if r['semantic_adequacy'] == 'adequate')}\n")
        handle.write(f"inadequate\t{sum(1 for r in rows if r['semantic_adequacy'] == 'inadequate')}\n")
        for reason, count in sorted(first_failures.items()):
            handle.write(f"first_failure.{reason}\t{count}\n")
    generated_at = datetime.now().isoformat(timespec="seconds")
    (out_dir / "prompt_summary_semantic_adequacy_metadata.json").write_text(json.dumps({
        "generated_by": "audit_prompt_summary_semantic_adequacy_v1.py",
        "generated_at": generated_at,
        "diagnostic_only": True,
        "benchmark_valid": "no",
        "evidence_root": str(evidence_root),
        "semantics": "semantic adequacy audit; does not require full numeric rows; not value authority",
    }, indent=2) + "\n", encoding="utf-8")
    (out_dir / "RUN_CONTEXT.md").write_text(
        "# prompt_summary_semantic_adequacy_diagnostic\n\n"
        "- diagnostic_only: yes\n- benchmark_valid: no\n"
        f"- generated_at: {generated_at}\n"
        f"- evidence_root: {evidence_root}\n"
        "- boundary: prompt semantic adequacy only; no LLM calls; full numeric row completeness is not required.\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    write_prompt_summary_audit(args.evidence_root, args.out_dir, args.repo_root)
    print(f"wrote prompt summary semantic adequacy audit to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
