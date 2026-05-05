#!/usr/bin/env python3
from __future__ import annotations

"""
S5-3b live LLM submission boundary for Stage5 direct-value candidates.

This supporting runner consumes frozen S5-3a prompt artifacts, loads API
credentials from the repo .env/environment through the existing Gemini helper,
persists raw responses and request metadata, parses returned JSON into S5-3
candidate TSV sidecars, and stops before S5-4 authority validation.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import call_gemini_stream_collect
    from src.utils.model_policy import PRIMARY_DEFAULT, validate_models_or_raise
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import call_gemini_stream_collect
    from src.utils.model_policy import PRIMARY_DEFAULT, validate_models_or_raise

ENTRYPOINT = "src/stage5_benchmark/run_s5_3b_llm_value_call_v1.py"
STAGE_BOUNDARY = "S5-3b"
BOUNDARY_CLASS = "diagnostic/supporting_boundary"
BENCHMARK_VALID_STATUS = "no"

CANDIDATE_TSV_NAME = "s5_3_llm_direct_value_candidates_v1.tsv"
EVIDENCE_SIDECAR_TSV_NAME = "s5_3_llm_direct_value_evidence_sidecar_v1.tsv"
REQUEST_SUMMARY_TSV_NAME = "s5_3b_llm_request_summary_v1.tsv"
INPUT_MANIFEST_TSV_NAME = "s5_3b_llm_input_manifest_v1.tsv"
RAW_RESPONSES_DIR_NAME = "s5_3_llm_raw_responses"
REQUEST_METADATA_DIR_NAME = "s5_3b_request_metadata"
RUN_CONTEXT_NAME = "RUN_CONTEXT.md"

RAW_RESPONSE_FILENAME_TEMPLATE = "{prompt_file_stem}__s5_3b_raw_response.json"
REQUEST_METADATA_FILENAME_TEMPLATE = "{prompt_file_stem}__s5_3b_request_metadata.json"

CANDIDATE_COLUMNS = [
    "paper_key",
    "formulation_id",
    "field_name",
    "value_text",
    "unit_text",
    "raw_cell_text",
    "direct_or_derived",
    "evidence_type",
    "evidence_scope",
    "source_file",
    "source_table_id",
    "source_row_id",
    "source_quote",
    "confidence",
    "needs_review",
    "llm_model",
    "prompt_hash",
    "input_artifact_hash",
    "llm_rationale_short",
]

EVIDENCE_COLUMNS = [
    "paper_key",
    "formulation_id",
    "field_name",
    "source_file",
    "source_table_id",
    "source_row_id",
    "source_quote",
    "raw_cell_text",
    "evidence_type",
    "evidence_scope",
    "direct_or_derived",
    "prompt_hash",
    "raw_response_path",
]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_existing_path(value: Path, role: str) -> Path:
    path = value.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required explicit {role} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Required explicit {role} is not a file: {path}")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError(f"JSONL row {line_no} is not an object: {path}")
            rows.append(parsed)
    return rows


def write_tsv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _clean(row.get(column)) for column in columns})


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_response_json(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    text = strip_code_fence(raw_text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, ""
        return None, "top_level_json_not_object"
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed, "recovered_json_object_from_text"
            except json.JSONDecodeError as exc:
                return None, f"json_decode_error_after_recovery:{exc}"
        return None, "json_decode_error"


def normalize_candidate(row: dict[str, Any], *, prompt_row: dict[str, Any], model: str, raw_response_path: Path) -> dict[str, str]:
    out = {column: _clean(row.get(column)) for column in CANDIDATE_COLUMNS}
    out["paper_key"] = out["paper_key"] or _clean(prompt_row.get("paper_key"))
    out["llm_model"] = model
    out["prompt_hash"] = _clean(prompt_row.get("prompt_sha256")) or sha256_text(_clean(prompt_row.get("prompt_text")))
    out["input_artifact_hash"] = _clean(prompt_row.get("input_artifact_hash"))
    if not out["needs_review"]:
        out["needs_review"] = "yes" if out["direct_or_derived"].lower() != "direct" or out["evidence_scope"].lower() in {"ambiguous", "unknown"} else "no"
    return out


def prompt_file_stem(row: dict[str, Any]) -> str:
    value = _clean(row.get("prompt_id")) or _clean(row.get("paper_key")) or "prompt"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "prompt"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen S5-3b live LLM submission from S5-3a prompt JSONL.")
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=PRIMARY_DEFAULT)
    parser.add_argument("--request-timeout-seconds", type=int, default=180)
    parser.add_argument("--request-retries", type=int, default=0)
    parser.add_argument("--retry-sleep-sec", type=float, default=3.0)
    parser.add_argument("--paper-key", action="append", default=[], dest="paper_keys")
    return parser


def filter_prompt_rows(rows: list[dict[str, Any]], requested: list[str]) -> list[dict[str, Any]]:
    requested_set = {x.strip() for x in requested if x.strip()}
    if not requested_set:
        return rows
    selected = [row for row in rows if _clean(row.get("paper_key")) in requested_set]
    found = {_clean(row.get("paper_key")) for row in selected}
    missing = sorted(requested_set - found)
    if missing:
        raise ValueError(f"Requested paper_key values not present in prompts_jsonl: {missing}")
    return selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_models_or_raise([args.model], context="S5-3b live-call runner")
    prompts_jsonl = resolve_existing_path(args.prompts_jsonl, "prompts-jsonl")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / RAW_RESPONSES_DIR_NAME
    metadata_dir = out_dir / REQUEST_METADATA_DIR_NAME
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    prompt_rows = filter_prompt_rows(read_jsonl(prompts_jsonl), args.paper_keys)
    if not prompt_rows:
        raise ValueError("No prompt rows selected for S5-3b live call")

    all_candidates: list[dict[str, str]] = []
    all_evidence: list[dict[str, str]] = []
    summary_rows: list[dict[str, Any]] = []
    for index, row in enumerate(prompt_rows, start=1):
        paper_key = _clean(row.get("paper_key"))
        prompt_text = _clean(row.get("prompt_text"))
        if not paper_key or not prompt_text:
            raise ValueError(f"Prompt row {index} missing paper_key or prompt_text")
        prompt_hash = _clean(row.get("prompt_sha256")) or sha256_text(prompt_text)
        file_stem = prompt_file_stem(row)
        raw_response_path = raw_dir / RAW_RESPONSE_FILENAME_TEMPLATE.format(prompt_file_stem=file_stem)
        request_metadata_path = metadata_dir / REQUEST_METADATA_FILENAME_TEMPLATE.format(prompt_file_stem=file_stem)
        started = datetime.now(timezone.utc).isoformat()
        metadata: dict[str, Any] = {
            "stage_boundary": STAGE_BOUNDARY,
            "paper_key": paper_key,
            "llm_backend": "gemini",
            "model": args.model,
            "generation_config": {"temperature": 0, "response_mime_type": "application/json"},
            "request_timeout_seconds": args.request_timeout_seconds,
            "request_retries": args.request_retries,
            "retry_sleep_sec": args.retry_sleep_sec,
            "source_prompts_jsonl_path": str(prompts_jsonl),
            "source_prompt_sha256": prompt_hash,
            "input_artifact_hash": _clean(row.get("input_artifact_hash")),
            "request_started_at_utc": started,
            "raw_response_path": str(raw_response_path),
            "api_key_source": "repo .env/environment via existing Gemini helper; secret value not recorded",
        }
        request_metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        status = "request_failure"
        raw_text = ""
        parse_status = "not_parsed"
        failure_type = ""
        failure_message = ""
        try:
            call_result = call_gemini_stream_collect(
                args.model,
                prompt_text,
                args.request_retries,
                args.retry_sleep_sec,
                progress_label=f"[{index}/{len(prompt_rows)}] {paper_key}",
                timeout_seconds=max(1, args.request_timeout_seconds),
            )
            raw_text = _clean(call_result.get("text"))
            status = _clean(call_result.get("status")) or "request_failure"
            if raw_text:
                raw_response_path.write_text(raw_text, encoding="utf-8")
            parsed, parse_error = parse_response_json(raw_text)
            if parsed is not None:
                candidate_objs = parsed.get("candidates", [])
                if isinstance(candidate_objs, list):
                    for candidate_obj in candidate_objs:
                        if not isinstance(candidate_obj, dict):
                            continue
                        candidate = normalize_candidate(candidate_obj, prompt_row=row, model=args.model, raw_response_path=raw_response_path)
                        all_candidates.append(candidate)
                        all_evidence.append(
                            {
                                **{field: candidate.get(field, "") for field in EVIDENCE_COLUMNS if field != "raw_response_path"},
                                "raw_response_path": str(raw_response_path),
                            }
                        )
                    parse_status = "parsed_candidates"
                else:
                    parse_status = "candidates_not_list"
            else:
                parse_status = parse_error
            metadata.update({k: v for k, v in call_result.items() if k != "text"})
        except Exception as exc:  # pragma: no cover - live API/transport failure path
            failure_type = type(exc).__name__
            failure_message = str(exc)
            metadata["api_failure"] = {"error_type": failure_type, "error_message": failure_message}
        finished = datetime.now(timezone.utc).isoformat()
        metadata.update(
            {
                "status": status,
                "request_finished_at_utc": finished,
                "raw_payload_persisted": bool(raw_text and raw_response_path.exists()),
                "raw_payload_character_count": len(raw_text),
                "raw_payload_sha256": sha256_text(raw_text) if raw_text else "",
                "parse_status": parse_status,
                "candidate_rows_after_parse": len([c for c in all_candidates if c.get("paper_key") == paper_key]),
            }
        )
        request_metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary_rows.append(
            {
                "paper_key": paper_key,
                "status": status,
                "model": args.model,
                "prompt_sha256": prompt_hash,
                "raw_response_path": raw_response_path if raw_response_path.exists() else "",
                "request_metadata_path": request_metadata_path,
                "raw_payload_persisted": "yes" if raw_text and raw_response_path.exists() else "no",
                "raw_payload_character_count": len(raw_text),
                "parse_status": parse_status,
                "candidate_rows_after_parse": len([c for c in all_candidates if c.get("paper_key") == paper_key]),
                "api_failure_type": failure_type,
                "api_failure_message": failure_message,
            }
        )

    candidate_tsv = out_dir / CANDIDATE_TSV_NAME
    evidence_tsv = out_dir / EVIDENCE_SIDECAR_TSV_NAME
    summary_tsv = out_dir / REQUEST_SUMMARY_TSV_NAME
    input_manifest = out_dir / INPUT_MANIFEST_TSV_NAME
    run_context = out_dir / RUN_CONTEXT_NAME
    write_tsv(candidate_tsv, CANDIDATE_COLUMNS, all_candidates)
    write_tsv(evidence_tsv, EVIDENCE_COLUMNS, all_evidence)
    write_tsv(
        summary_tsv,
        [
            "paper_key",
            "status",
            "model",
            "prompt_sha256",
            "raw_response_path",
            "request_metadata_path",
            "raw_payload_persisted",
            "raw_payload_character_count",
            "parse_status",
            "candidate_rows_after_parse",
            "api_failure_type",
            "api_failure_message",
        ],
        summary_rows,
    )
    write_tsv(
        input_manifest,
        ["input_role", "path", "exists", "sha256_or_metadata"],
        [
            {"input_role": "prompts_jsonl", "path": prompts_jsonl, "exists": "yes", "sha256_or_metadata": sha256_file(prompts_jsonl)},
            {"input_role": "raw_responses_dir", "path": raw_dir, "exists": "yes", "sha256_or_metadata": "directory"},
            {"input_role": "request_metadata_dir", "path": metadata_dir, "exists": "yes", "sha256_or_metadata": "directory"},
        ],
    )
    success_count = sum(1 for row in summary_rows if row.get("status") == "success")
    failure_count = len(summary_rows) - success_count
    run_context.write_text(
        "\n".join(
            [
                "# RUN_CONTEXT",
                "",
                "## Entrypoint",
                f"- entrypoint: `{ENTRYPOINT}`",
                f"- stage_boundary: `{STAGE_BOUNDARY}`",
                f"- boundary_class: `{BOUNDARY_CLASS}`",
                "",
                "## Benchmark-valid status",
                f"- benchmark_valid_status: `{BENCHMARK_VALID_STATUS}`",
                "- benchmark_valid: `no`",
                "- reason: `S5-3b persists raw LLM responses and candidate sidecars only; it is not benchmark scoring or final-table authority`",
                "",
                "## Exact inputs",
                f"- prompts_jsonl: `{prompts_jsonl}`",
                "",
                "## Live call settings",
                "- llm_backend: `gemini`",
                f"- model: `{args.model}`",
                "- API credentials: loaded from repo `.env` / environment by existing Gemini helper; key value not recorded or hard-coded",
                f"- request_timeout_seconds: `{args.request_timeout_seconds}`",
                f"- request_retries: `{args.request_retries}`",
                "",
                "## Outputs",
                f"- candidate_tsv: `{candidate_tsv}`",
                f"- evidence_sidecar_tsv: `{evidence_tsv}`",
                f"- raw_responses_dir: `{raw_dir}`",
                f"- request_metadata_dir: `{metadata_dir}`",
                f"- request_summary_tsv: `{summary_tsv}`",
                f"- input_manifest_tsv: `{input_manifest}`",
                "",
                "## Stop rule",
                "- This run stops before S5-4 value authority validation/merge.",
                "- Candidate rows are proposals only and must be validated by S5-4 before use.",
                "",
                "## Summary",
                f"- request_count: `{len(summary_rows)}`",
                f"- success_count: `{success_count}`",
                f"- failure_count: `{failure_count}`",
                f"- candidate_rows: `{len(all_candidates)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "status": "success" if failure_count == 0 else "partial_failure",
        "stage_boundary": STAGE_BOUNDARY,
        "out_dir": str(out_dir),
        "request_count": len(summary_rows),
        "success_count": success_count,
        "failure_count": failure_count,
        "candidate_rows": len(all_candidates),
        "outputs": {
            "candidate_tsv": str(candidate_tsv),
            "evidence_sidecar_tsv": str(evidence_tsv),
            "raw_responses_dir": str(raw_dir),
            "request_metadata_dir": str(metadata_dir),
            "request_summary_tsv": str(summary_tsv),
            "input_manifest_tsv": str(input_manifest),
            "run_context_md": str(run_context),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
