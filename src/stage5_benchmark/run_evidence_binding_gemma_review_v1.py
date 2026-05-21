#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
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


CORE_VALUE_FIELDS = {
    "la_ga_ratio_value",
    "polymer_mw_kDa_value",
    "plga_mass_mg_value",
    "surfactant_concentration_text_value",
    "organic_solvent_value",
    "drug_feed_amount_text_value",
    "size_nm_value",
    "pdi_value",
    "zeta_mV_value",
    "encapsulation_efficiency_percent_value",
    "loading_content_percent_value",
    "dl_percent_value",
    "organic_phase_volume_mL_value",
    "external_aqueous_phase_volume_mL_value",
    "surfactant_concentration_value_value",
    "surfactant_concentration_unit_value",
}

FIELD_REVIEW_STATUSES = {
    "evidence_binding_supported",
    "replacement_evidence_found",
    "unresolved_evidence_defect",
    "possible_value_error",
    "needs_human_review",
}

FORMULATION_VERDICTS = {
    "all_high_risk_fields_resolved",
    "some_fields_unresolved",
    "possible_value_error",
    "needs_human_review",
}

SUMMARY_FIELDS = [
    "paper_key",
    "final_formulation_id",
    "row_risk_level",
    "review_priority",
    "audit_model",
    "audit_status",
    "formulation_verdict",
    "field_review_count",
    "replacement_evidence_count",
    "unresolved_evidence_defect_count",
    "possible_value_error_count",
    "needs_human_review_count",
    "reviewed_fields",
    "recommended_next_action",
    "reason",
    "prompt_path",
    "raw_response_path",
]

FIELD_FIELDS = [
    "paper_key",
    "final_formulation_id",
    "field_name",
    "frozen_value",
    "risk_level",
    "evidence_status",
    "gemma_field_status",
    "replacement_evidence_text",
    "replacement_evidence_source",
    "reason",
    "recommended_next_action",
    "audit_model",
    "audit_status",
    "raw_response_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a diagnostic Gemma/Gemini review over Evidence Binding high-risk formulation rows. "
            "The reviewer sees only system-built evidence packs, never the full paper, and writes sidecar outputs only."
        )
    )
    parser.add_argument("--evidence-binding-packs-jsonl", required=True)
    parser.add_argument("--field-risk-tsv", required=True)
    parser.add_argument("--row-review-queue-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="gemma-4-31b-it")
    parser.add_argument("--risk-level", action="append", default=["high"])
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--request-timeout-seconds", type=int, default=180)
    parser.add_argument("--request-retries", type=int, default=0)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--field-scope",
        choices=["high_only", "high_and_medium"],
        default="high_and_medium",
        help="Which row-local risky fields to include as audit targets. Context fields may still be shown.",
    )
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


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def compact(text: Any, limit: int = 1200) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def safe_name(*parts: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", "__".join(parts))[:180]


def load_tsv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: scalar(row.get(field, "")) for field in fields})


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_existing_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = load_tsv(path)
    return {row.get("final_formulation_id", ""): row for row in rows if row.get("final_formulation_id")}


def load_packs_for_formulations(path: Path, selected_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    packs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            fid = str(payload.get("final_formulation_id", ""))
            if fid in selected_ids:
                packs[fid].append(payload)
    return packs


def risk_fields_by_formulation(rows: list[dict[str, str]], *, field_scope: str) -> dict[str, list[dict[str, str]]]:
    allowed_risks = {"high"} if field_scope == "high_only" else {"high", "medium"}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("risk_level") not in allowed_risks:
            continue
        if row.get("frozen_value_present") != "yes":
            continue
        if row.get("field_name") not in CORE_VALUE_FIELDS:
            continue
        grouped[row["final_formulation_id"]].append(row)
    return grouped


def build_prompt(
    *,
    row: dict[str, str],
    target_fields: list[dict[str, str]],
    packs: list[dict[str, Any]],
) -> str:
    pack_by_field = {str(pack.get("field_name", "")): pack for pack in packs}
    target_payloads: list[dict[str, Any]] = []
    for risk in target_fields:
        field_name = risk.get("field_name", "")
        pack = pack_by_field.get(field_name, {})
        target_payloads.append(
            {
                "field_name": field_name,
                "frozen_value": pack.get("frozen_value") or risk.get("review_display_text", ""),
                "risk_level": risk.get("risk_level", ""),
                "risk_reason": risk.get("reason", ""),
                "evidence_status": risk.get("evidence_status", ""),
                "evidence_contains_exact_value": risk.get("evidence_contains_exact_value", ""),
                "value_evidence_text": compact(pack.get("value_evidence_text", ""), 900),
                "row_identity_evidence_text": compact(pack.get("row_identity_evidence_text", ""), 900),
                "source_cell_text": compact(pack.get("source_cell_text", ""), 900),
                "source_row_label": compact(pack.get("source_row_label", ""), 300),
                "source_column_label": compact(pack.get("source_column_label", ""), 300),
                "source_locator_text": risk.get("source_locator_text", ""),
                "review_display_text": risk.get("review_display_text", ""),
            }
        )

    context_payloads: list[dict[str, Any]] = []
    for pack in packs:
        field_name = str(pack.get("field_name", ""))
        if field_name in {item["field_name"] for item in target_payloads}:
            continue
        if field_name not in CORE_VALUE_FIELDS and field_name != "polymer_identity_final":
            continue
        status = str(pack.get("binding_status", ""))
        frozen_value = str(pack.get("frozen_value", ""))
        if not frozen_value or status == "blank_value":
            continue
        context_payloads.append(
            {
                "field_name": field_name,
                "frozen_value": compact(frozen_value, 300),
                "binding_status": status,
                "value_evidence_text": compact(pack.get("value_evidence_text", ""), 500),
                "row_identity_evidence_text": compact(pack.get("row_identity_evidence_text", ""), 500),
                "source_cell_text": compact(pack.get("source_cell_text", ""), 500),
                "source_locator_text": compact(pack.get("source_locator_text", ""), 300),
            }
        )
    context_payloads = context_payloads[:24]

    return "\n".join(
        [
            "You are auditing Evidence Binding sidecar data for one PLGA formulation row.",
            "You must not read or infer from the full paper. Use only the supplied system evidence package.",
            "You must not create formulations, remove formulations, or change frozen values.",
            "Your job is to judge whether each target field's evidence text/cell supports its frozen value,",
            "and whether a better replacement evidence snippet is already present inside this evidence package.",
            "",
            "For each target field, choose exactly one field_status:",
            "- evidence_binding_supported: current evidence text/cell supports the frozen value.",
            "- replacement_evidence_found: current value evidence is bad/missing, but another supplied snippet in this package contains the frozen value and can replace it.",
            "- unresolved_evidence_defect: supplied evidence package does not contain evidence supporting the frozen value.",
            "- possible_value_error: supplied evidence suggests the frozen value itself may be wrong, but do not correct it.",
            "- needs_human_review: cannot decide from the package.",
            "",
            "Return only JSON with this schema:",
            "{",
            '  "formulation_verdict": "all_high_risk_fields_resolved|some_fields_unresolved|possible_value_error|needs_human_review",',
            '  "reason": "one concise sentence",',
            '  "recommended_next_action": "accept_evidence|replace_evidence_sidecar|human_review|investigate_value",',
            '  "field_reviews": [',
            '    {"field_name":"...", "field_status":"...", "replacement_evidence_text":"", "replacement_evidence_source":"", "reason":"one short sentence", "recommended_next_action":"..."}',
            "  ]",
            "}",
            "",
            "Formulation row:",
            json.dumps(
                {
                    "paper_key": row.get("paper_key", ""),
                    "final_formulation_id": row.get("final_formulation_id", ""),
                    "row_risk_level": row.get("row_risk_level", ""),
                    "review_priority": row.get("review_priority", ""),
                    "row_risk_reasons": row.get("row_risk_reasons", ""),
                    "suggested_review_focus": row.get("suggested_review_focus", ""),
                    "core_fields_present": row.get("core_fields_present", ""),
                    "core_fields_high_risk": row.get("core_fields_high_risk", ""),
                    "core_fields_medium_risk": row.get("core_fields_medium_risk", ""),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "Target fields to audit:",
            json.dumps(target_payloads, ensure_ascii=False, indent=2),
            "",
            "Additional same-formulation evidence context:",
            json.dumps(context_payloads, ensure_ascii=False, indent=2),
        ]
    )


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        for match in re.finditer(r"\{", stripped):
            try:
                payload, _ = decoder.raw_decode(stripped[match.start() :])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise
    if not isinstance(payload, dict):
        raise ValueError("Reviewer response was not a JSON object.")
    return payload


def normalize_review(payload: dict[str, Any], target_fields: list[dict[str, str]]) -> dict[str, Any]:
    verdict = str(payload.get("formulation_verdict", "")).strip()
    if verdict not in FORMULATION_VERDICTS:
        verdict = "needs_human_review"
    by_field: dict[str, dict[str, Any]] = {}
    for item in payload.get("field_reviews", []):
        if isinstance(item, dict):
            by_field[str(item.get("field_name", ""))] = item
    field_reviews: list[dict[str, str]] = []
    for target in target_fields:
        field_name = target.get("field_name", "")
        item = by_field.get(field_name, {})
        status = str(item.get("field_status", "")).strip()
        if status not in FIELD_REVIEW_STATUSES:
            status = "needs_human_review"
        field_reviews.append(
            {
                "field_name": field_name,
                "field_status": status,
                "replacement_evidence_text": str(item.get("replacement_evidence_text", "")).strip(),
                "replacement_evidence_source": str(item.get("replacement_evidence_source", "")).strip(),
                "reason": str(item.get("reason", "")).strip(),
                "recommended_next_action": str(item.get("recommended_next_action", "")).strip(),
            }
        )
    return {
        "audit_status": "parsed",
        "formulation_verdict": verdict,
        "reason": str(payload.get("reason", "")).strip(),
        "recommended_next_action": str(payload.get("recommended_next_action", "")).strip(),
        "field_reviews": field_reviews,
    }


def failed_review(error: str, target_fields: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "audit_status": "error",
        "formulation_verdict": "needs_human_review",
        "reason": error,
        "recommended_next_action": "human_review",
        "field_reviews": [
            {
                "field_name": row.get("field_name", ""),
                "field_status": "needs_human_review",
                "replacement_evidence_text": "",
                "replacement_evidence_source": "",
                "reason": error,
                "recommended_next_action": "human_review",
            }
            for row in target_fields
        ],
    }


def summarize_field_reviews(field_reviews: list[dict[str, str]]) -> dict[str, Any]:
    counts = Counter(row.get("field_status", "") for row in field_reviews)
    return {
        "field_review_count": len(field_reviews),
        "replacement_evidence_count": counts.get("replacement_evidence_found", 0),
        "unresolved_evidence_defect_count": counts.get("unresolved_evidence_defect", 0),
        "possible_value_error_count": counts.get("possible_value_error", 0),
        "needs_human_review_count": counts.get("needs_human_review", 0),
        "reviewed_fields": ";".join(row.get("field_name", "") for row in field_reviews),
    }


def write_run_context(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    selected_count: int,
    completed_count: int,
    summary_counts: Counter[str],
    field_counts: Counter[str],
) -> None:
    lines = [
        "# RUN_CONTEXT",
        "",
        "## Run purpose",
        "Run Gemma/Gemini diagnostic review over Evidence Binding high-risk formulation rows.",
        "",
        "## Run type",
        "Diagnostic-only evidence-review sidecar. Not benchmark-valid final output.",
        "",
        "## Boundary",
        "This run consumes frozen Evidence Binding Packs and risk queues only. It does not read full papers, create rows, create values, mutate packs, mutate the Stage5 final table, or update ACTIVE_RUN.json.",
        "",
        "## Inputs",
        f"- evidence_binding_packs_jsonl: `{to_repo_rel(repo_path(args.evidence_binding_packs_jsonl))}`",
        f"- field_risk_tsv: `{to_repo_rel(repo_path(args.field_risk_tsv))}`",
        f"- row_review_queue_tsv: `{to_repo_rel(repo_path(args.row_review_queue_tsv))}`",
        "",
        "## Model",
        f"- model: `{args.model}`",
        "- backend: `gemini_stream_collect`",
        "",
        "## Outputs",
        "- `analysis/gemma_evidence_binding_formulation_reviews_v1.tsv`",
        "- `analysis/gemma_evidence_binding_field_reviews_v1.tsv`",
        "- `analysis/gemma_evidence_binding_review_details_v1.jsonl`",
        "- `prompts/*.txt`",
        "- `raw_responses/*.json`",
        "",
        "## Counts",
        f"- selected_formulation_rows: {selected_count}",
        f"- completed_formulation_rows: {completed_count}",
        "",
        "## Formulation verdict distribution",
    ]
    for key, value in sorted(summary_counts.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Field review status distribution"])
    for key, value in sorted(field_counts.items()):
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append(f"generated_at: `{now_iso()}`")
    (out_dir / "RUN_CONTEXT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = repo_path(args.out_dir)
    for subdir in ["analysis", "prompts", "raw_responses"]:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "analysis" / "gemma_evidence_binding_formulation_reviews_v1.tsv"
    field_path = out_dir / "analysis" / "gemma_evidence_binding_field_reviews_v1.tsv"
    detail_path = out_dir / "analysis" / "gemma_evidence_binding_review_details_v1.jsonl"
    if args.overwrite:
        for path in [summary_path, field_path, detail_path]:
            if path.exists():
                path.unlink()

    existing = read_existing_summary(summary_path)
    row_queue = load_tsv(repo_path(args.row_review_queue_tsv))
    field_risk = load_tsv(repo_path(args.field_risk_tsv))
    target_by_fid = risk_fields_by_formulation(field_risk, field_scope=args.field_scope)
    allowed_risks = {value.lower() for value in args.risk_level}
    selected = [
        row
        for row in row_queue
        if row.get("row_risk_level", "").lower() in allowed_risks and row.get("final_formulation_id", "") in target_by_fid
    ]
    selected.sort(key=lambda row: (int(row.get("review_priority") or 0) * -1, row.get("paper_key", ""), row.get("final_formulation_id", "")))
    if args.max_rows > 0:
        selected = selected[: args.max_rows]
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must be in [0, shard-count)")
    if args.shard_count > 1:
        selected = [row for index, row in enumerate(selected) if index % args.shard_count == args.shard_index]

    selected_ids = {row["final_formulation_id"] for row in selected}
    packs_by_fid = load_packs_for_formulations(repo_path(args.evidence_binding_packs_jsonl), selected_ids)

    summary_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = load_tsv(field_path) if field_path.exists() and not args.overwrite else []
    if existing:
        summary_rows.extend(existing.values())

    completed = len(existing)
    for index, row in enumerate(selected, start=1):
        fid = row["final_formulation_id"]
        if fid in existing:
            continue
        target_fields = target_by_fid[fid]
        prompt = build_prompt(row=row, target_fields=target_fields, packs=packs_by_fid.get(fid, []))
        name = safe_name(row.get("paper_key", ""), fid)
        prompt_path = out_dir / "prompts" / f"{name}__gemma_evidence_review_prompt.txt"
        raw_path = out_dir / "raw_responses" / f"{name}__gemma_evidence_review_raw_response.json"
        prompt_path.write_text(prompt, encoding="utf-8")

        text = ""
        result: dict[str, Any] = {}
        try:
            result = call_gemini_stream_collect(
                args.model,
                prompt,
                retries=max(0, args.request_retries),
                sleep_sec=max(0.0, args.retry_sleep_seconds),
                progress_label=f"[{index}/{len(selected)}] {row.get('paper_key')} {fid}",
                timeout_seconds=args.request_timeout_seconds,
            )
            text = str(result.get("text", "") or "")
            if result.get("status") != "success":
                raise RuntimeError(str(result.get("error_message") or result.get("status") or "Gemma request failed"))
            normalized = normalize_review(extract_json_object(text), target_fields)
            raw_payload = {"row": row, "target_fields": target_fields, "model": args.model, "call_result": result, "content": text}
        except Exception as exc:  # noqa: BLE001
            normalized = failed_review(str(exc), target_fields)
            raw_payload = {"row": row, "target_fields": target_fields, "model": args.model, "call_result": result, "error": str(exc), "content": text}

        raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        field_reviews = normalized["field_reviews"]
        counts = summarize_field_reviews(field_reviews)
        summary_row = {
            "paper_key": row.get("paper_key", ""),
            "final_formulation_id": fid,
            "row_risk_level": row.get("row_risk_level", ""),
            "review_priority": row.get("review_priority", ""),
            "audit_model": args.model,
            "audit_status": normalized["audit_status"],
            "formulation_verdict": normalized["formulation_verdict"],
            **counts,
            "recommended_next_action": normalized["recommended_next_action"],
            "reason": normalized["reason"],
            "prompt_path": to_repo_rel(prompt_path),
            "raw_response_path": to_repo_rel(raw_path),
        }
        summary_rows.append(summary_row)
        for review in field_reviews:
            risk = next((item for item in target_fields if item.get("field_name") == review.get("field_name")), {})
            pack = next((item for item in packs_by_fid.get(fid, []) if item.get("field_name") == review.get("field_name")), {})
            field_rows.append(
                {
                    "paper_key": row.get("paper_key", ""),
                    "final_formulation_id": fid,
                    "field_name": review.get("field_name", ""),
                    "frozen_value": pack.get("frozen_value", ""),
                    "risk_level": risk.get("risk_level", ""),
                    "evidence_status": risk.get("evidence_status", ""),
                    "gemma_field_status": review.get("field_status", ""),
                    "replacement_evidence_text": review.get("replacement_evidence_text", ""),
                    "replacement_evidence_source": review.get("replacement_evidence_source", ""),
                    "reason": review.get("reason", ""),
                    "recommended_next_action": review.get("recommended_next_action", ""),
                    "audit_model": args.model,
                    "audit_status": normalized["audit_status"],
                    "raw_response_path": to_repo_rel(raw_path),
                }
            )
        append_jsonl(detail_path, {"summary": summary_row, "field_reviews": field_reviews})
        completed += 1
        write_tsv(summary_path, summary_rows, SUMMARY_FIELDS)
        write_tsv(field_path, field_rows, FIELD_FIELDS)
        print(
            f"[{index}/{len(selected)}] completed {row.get('paper_key')} {fid}: "
            f"{normalized['audit_status']} {normalized['formulation_verdict']}",
            flush=True,
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    summary_counts = Counter(str(row.get("formulation_verdict", "")) for row in summary_rows)
    field_counts = Counter(str(row.get("gemma_field_status", "")) for row in field_rows)
    write_run_context(
        out_dir,
        args=args,
        selected_count=len(selected),
        completed_count=completed,
        summary_counts=summary_counts,
        field_counts=field_counts,
    )
    print(f"selected_formulation_rows\t{len(selected)}")
    print(f"completed_formulation_rows\t{completed}")
    print(f"out_dir\t{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
