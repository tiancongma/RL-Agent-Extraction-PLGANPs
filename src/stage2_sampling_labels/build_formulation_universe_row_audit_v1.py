#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import call_gemini_stream_collect
    from src.utils.paths import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import call_gemini_stream_collect
    from src.utils.paths import PROJECT_ROOT


VERDICTS = {
    "supported",
    "weakly_supported",
    "unsupported",
    "duplicate_or_alias",
    "control_not_prepared_formulation",
    "needs_human_review",
}

OUT_FIELDS = [
    "paper_key",
    "canonical_formulation_id",
    "formulation_label",
    "row_role",
    "risk_level",
    "risk_flags_json",
    "audit_model",
    "audit_status",
    "verdict",
    "reason",
    "evidence_issue",
    "recommended_action",
    "raw_response_path",
    "prompt_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run row-level formulation evidence audit with a reviewer LLM.")
    parser.add_argument("--formulation-run-dir", required=True)
    parser.add_argument("--risk-flags-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--risk-level", action="append", default=["HIGH"])
    parser.add_argument("--model", default="gemma-4-31b-it")
    parser.add_argument("--request-timeout-seconds", type=int, default=180)
    parser.add_argument("--request-retries", type=int, default=0)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def load_tsv(path: Path) -> list[dict[str, str]]:
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


def normalize_row_id(row: dict[str, str]) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{row['paper_key']}__{row['canonical_formulation_id']}__{row['formulation_label']}")[:180]


def compact_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def context_for_quote(prompt_text: str, quote: str, *, radius: int = 1200) -> str:
    prompt = prompt_text or ""
    marker = "Full/source text:"
    if marker in prompt:
        prompt = prompt.split(marker, 1)[1]
    q = compact_space(quote)
    if not prompt:
        return ""
    candidates = [q]
    if len(q) > 160:
        candidates.append(q[:160])
    for candidate in candidates:
        if not candidate:
            continue
        index = compact_space(prompt).find(candidate)
        if index >= 0:
            compact = compact_space(prompt)
            start = max(0, index - radius)
            end = min(len(compact), index + len(candidate) + radius)
            return compact[start:end]
    return compact_space(prompt[:2400])


def build_prompt(row: dict[str, str], risk_row: dict[str, str], context: str) -> str:
    return "\n".join(
        [
            "You are auditing one proposed formulation row. Do not create new rows.",
            "Judge only whether this exact row is supported by the supplied evidence.",
            "",
            "Allowed verdict values:",
            "- supported",
            "- weakly_supported",
            "- unsupported",
            "- duplicate_or_alias",
            "- control_not_prepared_formulation",
            "- needs_human_review",
            "",
            "Return only this small JSON object:",
            '{"verdict":"supported|weakly_supported|unsupported|duplicate_or_alias|control_not_prepared_formulation|needs_human_review","reason":"one short sentence","evidence_issue":"none|missing_quote|quote_not_row_specific|locator_unclear|control_boundary|duplicate_boundary|other","recommended_action":"accept_row|review_row|exclude_row|merge_or_alias"}',
            "",
            "Proposed row:",
            f"paper_key: {row.get('paper_key', '')}",
            f"canonical_formulation_id: {row.get('canonical_formulation_id', '')}",
            f"formulation_label: {row.get('formulation_label', '')}",
            f"row_role: {row.get('row_role', '')}",
            f"identity_basis: {row.get('identity_basis', '')}",
            f"preparation_evidence_quote: {row.get('preparation_evidence_quote', '')}",
            f"source_locator: {row.get('source_locator', '')}",
            f"paper_risk_level: {risk_row.get('risk_level', '')}",
            f"paper_risk_flags: {risk_row.get('risk_flags_json', '')}",
            "",
            "Nearby source context:",
            context[:3000],
        ]
    )


def parse_verdict(text: str) -> dict[str, str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {"audit_status": "parse_error", "verdict": "needs_human_review", "reason": "Reviewer did not return JSON.", "evidence_issue": "other", "recommended_action": "review_row"}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"audit_status": "parse_error", "verdict": "needs_human_review", "reason": "Reviewer JSON was malformed.", "evidence_issue": "other", "recommended_action": "review_row"}
    verdict = str(payload.get("verdict", "")).strip()
    if verdict not in VERDICTS:
        verdict = "needs_human_review"
    return {
        "audit_status": "parsed",
        "verdict": verdict,
        "reason": str(payload.get("reason", "")).strip(),
        "evidence_issue": str(payload.get("evidence_issue", "")).strip(),
        "recommended_action": str(payload.get("recommended_action", "")).strip(),
    }


def main() -> int:
    args = parse_args()
    run_dir = repo_path(args.formulation_run_dir)
    out_dir = repo_path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory exists and is not empty: {out_dir}. Use --overwrite.")
    for subdir in ["prompts", "raw_responses", "analysis"]:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    rows = load_tsv(run_dir / "formulation_universe_frozen_v1.tsv")
    risk_rows = {row["paper_key"]: row for row in load_tsv(repo_path(args.risk_flags_tsv))}
    allowed_levels = set(args.risk_level)
    selected = [row for row in rows if risk_rows.get(row["paper_key"], {}).get("risk_level") in allowed_levels]
    if args.max_rows > 0:
        selected = selected[: args.max_rows]

    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        risk_row = risk_rows.get(row["paper_key"], {})
        prompt_path = run_dir / "prompts" / f"{row['paper_key']}__formulation_universe_prompt.txt"
        prompt_text = prompt_path.read_text(encoding="utf-8", errors="replace") if prompt_path.exists() else ""
        context = context_for_quote(prompt_text, row.get("preparation_evidence_quote", ""))
        audit_prompt = build_prompt(row, risk_row, context)
        row_id = normalize_row_id(row)
        out_prompt_path = out_dir / "prompts" / f"{row_id}__row_audit_prompt.txt"
        raw_path = out_dir / "raw_responses" / f"{row_id}__row_audit_raw_response.json"
        out_prompt_path.write_text(audit_prompt, encoding="utf-8")
        try:
            result = call_gemini_stream_collect(
                args.model,
                audit_prompt,
                retries=max(0, args.request_retries),
                sleep_sec=max(0.0, args.retry_sleep_seconds),
                progress_label="",
                timeout_seconds=args.request_timeout_seconds,
            )
            text = str(result.get("text", "") or "")
            if result.get("status") != "success":
                raise RuntimeError(str(result.get("error_message") or result.get("status") or "Gemma request failed"))
            parsed = parse_verdict(text)
            raw_path.write_text(
                json.dumps({"row": row, "model": args.model, "content": text, "call_result": result}, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            parsed = {"audit_status": "error", "verdict": "needs_human_review", "reason": str(exc), "evidence_issue": "other", "recommended_action": "review_row"}
            raw_path.write_text(
                json.dumps({"row": row, "model": args.model, "content": "", "error": str(exc)}, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        output_rows.append(
            {
                **row,
                "risk_level": risk_row.get("risk_level", ""),
                "risk_flags_json": risk_row.get("risk_flags_json", ""),
                "audit_model": args.model,
                **parsed,
                "raw_response_path": to_repo_rel(raw_path),
                "prompt_path": to_repo_rel(out_prompt_path),
            }
        )
        print(f"[{index}/{len(selected)}] {row['paper_key']} {row['canonical_formulation_id']} {row['formulation_label']}: {parsed['audit_status']} {parsed['verdict']}", flush=True)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    write_tsv(out_dir / "row_level_verdicts_v1.tsv", output_rows, OUT_FIELDS)
    (out_dir / "RUN_CONTEXT.md").write_text(
        "\n".join(
            [
                "# formulation_universe_row_audit_v1",
                "",
                f"generated_at: {now_iso()}",
                "generated_by: `src/stage2_sampling_labels/build_formulation_universe_row_audit_v1.py`",
                "benchmark_valid: `no`",
                "audit_layer: `row-level reviewer only`",
                "",
                "Boundary:",
                "- This reviewer does not create formulation rows.",
                "- This reviewer does not extract values.",
                "- Verdicts are diagnostic review labels over an existing formulation-universe artifact.",
                "",
                f"formulation_run_dir: `{to_repo_rel(run_dir)}`",
                f"risk_flags_tsv: `{to_repo_rel(repo_path(args.risk_flags_tsv))}`",
                f"model: `{args.model}`",
                f"selected_rows: `{len(selected)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {to_repo_rel(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
