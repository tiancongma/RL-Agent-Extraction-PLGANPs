#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.stage2_sampling_labels.ollama_local_backend_v1 import (
        call_ollama_chat_nonstream,
        resolve_ollama_base_url,
        resolve_ollama_model,
    )
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import resolve_results_write_target
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage2_sampling_labels.ollama_local_backend_v1 import (
        call_ollama_chat_nonstream,
        resolve_ollama_base_url,
        resolve_ollama_model,
    )
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import resolve_results_write_target


RUN_METADATA_NAME = "stage2_s2_4b_ollama_local_diagnostic_run_metadata_v1.json"
REQUEST_SUMMARY_NAME = "s2_4b_request_summary_v1.tsv"
RAW_RESPONSE_FILENAME_TEMPLATE = "{paper_key}__stage2_v2_raw_response.json"
RAW_RESPONSE_WRAPPER_SIDECAR_TEMPLATE = "{paper_key}__stage2_v2_raw_response.original_wrapper.json"
REQUEST_METADATA_FILENAME_TEMPLATE = "{paper_key}__stage2_v2_request_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a local-only diagnostic S2-4b-style Ollama call boundary from immutable S2-4a prompt artifacts and stop after raw-response persistence."
    )
    parser.add_argument("--run-id", default="", help="Explicit legacy compatibility run_id. New writes default to MDEC084 v2 bucket/child naming when omitted.")
    parser.add_argument("--run-dir", default="", help="Explicit results run directory. Supports legacy roots and v2 child execution paths under data/results/.")
    parser.add_argument("--execution-cue", default="s2_4b_ollama_local_diagnostic", help="Future-facing child cue used only when auto-allocating a new v2 child execution path.")
    parser.add_argument("--prompts-jsonl", required=True, help="Frozen S2-4a prompt JSONL artifact. Each row must contain paper_key and prompt_text.")
    parser.add_argument("--prompt-template", default="", help="Optional frozen S2-4a prompt template path recorded for traceability.")
    parser.add_argument("--prompt-audit", default="", help="Optional frozen S2-4a prompt audit TSV path recorded for traceability.")
    parser.add_argument("--freeze-manifest", default="", help="Optional FREEZE_MANIFEST.md path recorded for traceability.")
    parser.add_argument("--paper-key", action="append", dest="paper_keys", default=[], help="Repeatable paper key filter. Default: all prompt rows in --prompts-jsonl.")
    parser.add_argument("--base-url", default="", help="Local Ollama base URL. Default: OLLAMA_BASE_URL or http://127.0.0.1:11434.")
    parser.add_argument("--model", default="", help="Local Ollama model. Required unless OLLAMA_MODEL is set.")
    parser.add_argument("--request-timeout-seconds", type=int, default=180, help="Per-request timeout in seconds for the local diagnostic Ollama path.")
    parser.add_argument("--request-retries", type=int, default=0, help="Request-level retries for the local diagnostic Ollama path.")
    parser.add_argument("--retry-sleep-sec", type=float, default=3.0)
    parser.add_argument("--max-parallel-requests", type=int, default=1, help="Maximum concurrent local requests. Default preserves serial behavior.")
    parser.add_argument("--inter-request-sleep-seconds", type=float, default=0.0, help="Optional sleep between request submissions.")
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def optional_repo_path(value: str) -> Path | None:
    if not str(value).strip():
        return None
    return repo_path(value)


def to_repo_rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError(f"Prompt row {index} is not a JSON object: {path}")
            rows.append(parsed)
    return rows


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def filter_prompt_rows(rows: list[dict[str, Any]], requested_keys: list[str]) -> list[dict[str, Any]]:
    if not requested_keys:
        return rows
    requested = {key.strip() for key in requested_keys if key.strip()}
    filtered = [row for row in rows if str(row.get("paper_key", "")).strip() in requested]
    found = {str(row.get("paper_key", "")).strip() for row in filtered}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"Requested paper keys missing from prompts JSONL: {missing}")
    return filtered


def build_request_metadata_payload(
    *,
    paper_key: str,
    doi: str,
    base_url: str,
    model: str,
    request_timeout_seconds: int,
    request_retries: int,
    retry_sleep_sec: float,
    prompts_jsonl: Path,
    prompt_template: Path | None,
    prompt_audit: Path | None,
    freeze_manifest: Path | None,
    prompt_sha256: str,
    row: dict[str, Any],
    raw_response_path: Path,
) -> dict[str, Any]:
    return {
        "stage_boundary": "S2-4b_style_diagnostic",
        "paper_key": paper_key,
        "doi": doi,
        "status": "",
        "llm_backend": "ollama_local_diagnostic",
        "base_url": base_url,
        "model": model,
        "request_timeout_seconds": request_timeout_seconds,
        "request_retries": request_retries,
        "retry_sleep_sec": retry_sleep_sec,
        "generation_config": {"temperature": 0},
        "request_mode": "non_streaming",
        "source_prompts_jsonl_path": to_repo_rel(prompts_jsonl),
        "source_prompt_template_path": to_repo_rel(prompt_template),
        "source_prompt_audit_path": to_repo_rel(prompt_audit),
        "source_freeze_manifest_path": to_repo_rel(freeze_manifest),
        "source_prompt_sha256": prompt_sha256,
        "source_prompt_row_fields": {
            "paper_key": paper_key,
            "doi": doi,
            "title": str(row.get("title", "")).strip(),
            "evidence_blocks_path": str(row.get("evidence_blocks_path", "")).strip(),
            "source_evidence_artifact_path": str(row.get("source_evidence_artifact_path", "")).strip(),
            "ordered_block_order": row.get("ordered_block_order", []),
        },
        "request_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_response_path": to_repo_rel(raw_response_path),
        "raw_payload_status": "not_written_yet",
        "raw_payload_persisted": False,
        "raw_payload_character_count": 0,
        "raw_payload_sha256": "",
        "normalization_attempted": False,
        "normalization_applied": False,
        "normalization_reason": "not_applicable",
        "original_wrapper_preserved": False,
        "original_wrapper_path": "",
        "original_wrapper_character_count": 0,
        "original_wrapper_sha256": "",
    }


def normalize_ollama_raw_payload_for_replay(raw_text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "persisted_raw_text": raw_text,
        "normalization_attempted": False,
        "normalization_applied": False,
        "normalization_reason": "not_applicable",
        "original_wrapper_text": "",
    }
    if not raw_text:
        result["normalization_reason"] = "empty_response_text"
        return result

    try:
        parsed_top_level = json.loads(raw_text)
    except Exception as exc:
        result["normalization_reason"] = f"top_level_json_parse_failed:{type(exc).__name__}"
        return result

    if not isinstance(parsed_top_level, dict):
        result["normalization_reason"] = "top_level_payload_not_dict"
        return result

    message = parsed_top_level.get("message")
    if not isinstance(message, dict):
        result["normalization_reason"] = "message_wrapper_not_present"
        return result

    content = message.get("content")
    if not isinstance(content, str):
        result["normalization_reason"] = "message_content_not_string"
        return result

    result["normalization_attempted"] = True
    try:
        parsed_content = json.loads(content)
    except Exception as exc:
        result["normalization_reason"] = f"message_content_json_parse_failed:{type(exc).__name__}"
        return result

    if not isinstance(parsed_content, dict):
        result["normalization_reason"] = "message_content_json_not_dict"
        return result

    result["persisted_raw_text"] = json.dumps(parsed_content, ensure_ascii=False, indent=2) + "\n"
    result["normalization_applied"] = True
    result["normalization_reason"] = "unwrapped_message_content_json_dict"
    result["original_wrapper_text"] = raw_text
    return result


def process_prompt_row(
    *,
    index: int,
    total: int,
    row: dict[str, Any],
    base_url: str,
    model: str,
    request_timeout_seconds: int,
    request_retries: int,
    retry_sleep_sec: float,
    prompts_jsonl: Path,
    prompt_template: Path | None,
    prompt_audit: Path | None,
    freeze_manifest: Path | None,
    raw_dir: Path,
    metadata_dir: Path,
    metadata_write_lock: threading.Lock,
) -> dict[str, Any]:
    paper_key = str(row.get("paper_key", "")).strip()
    prompt_text = str(row.get("prompt_text", ""))
    doi = str(row.get("doi", "")).strip()
    if not paper_key:
        raise ValueError(f"Prompt row {index} is missing paper_key.")
    if not prompt_text:
        raise ValueError(f"Prompt row {index} / {paper_key} is missing prompt_text.")

    prompt_sha256 = str(row.get("prompt_sha256", "")).strip() or hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    raw_response_path = raw_dir / RAW_RESPONSE_FILENAME_TEMPLATE.format(paper_key=paper_key)
    raw_response_wrapper_path = raw_dir / RAW_RESPONSE_WRAPPER_SIDECAR_TEMPLATE.format(paper_key=paper_key)
    request_metadata_path = metadata_dir / REQUEST_METADATA_FILENAME_TEMPLATE.format(paper_key=paper_key)
    progress_label = f"[{index}/{total}] {paper_key}"
    print(
        f"{progress_label} sending_prompt source={to_repo_rel(prompts_jsonl)} prompt_sha256={prompt_sha256} "
        f"model={model} base_url={base_url}",
        flush=True,
    )

    metadata_payload = build_request_metadata_payload(
        paper_key=paper_key,
        doi=doi,
        base_url=base_url,
        model=model,
        request_timeout_seconds=request_timeout_seconds,
        request_retries=request_retries,
        retry_sleep_sec=retry_sleep_sec,
        prompts_jsonl=prompts_jsonl,
        prompt_template=prompt_template,
        prompt_audit=prompt_audit,
        freeze_manifest=freeze_manifest,
        prompt_sha256=prompt_sha256,
        row=row,
        raw_response_path=raw_response_path,
    )
    with metadata_write_lock:
        request_metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    raw_text = ""
    try:
        call_result = call_ollama_chat_nonstream(
            base_url=base_url,
            model=model,
            prompt=prompt_text,
            timeout_seconds=request_timeout_seconds,
            retries=request_retries,
            sleep_sec=retry_sleep_sec,
            progress_label=progress_label,
        )
        raw_text = str(call_result.get("text", "") or "")
        normalized_payload = normalize_ollama_raw_payload_for_replay(raw_text)
        persisted_raw_text = str(normalized_payload.get("persisted_raw_text", "") or "")
        if persisted_raw_text:
            raw_response_path.write_text(persisted_raw_text, encoding="utf-8")
        original_wrapper_text = str(normalized_payload.get("original_wrapper_text", "") or "")
        if original_wrapper_text:
            raw_response_wrapper_path.write_text(original_wrapper_text, encoding="utf-8")
        metadata_payload["raw_payload_persisted"] = bool(raw_text and raw_response_path.exists())
        metadata_payload["raw_payload_character_count"] = len(persisted_raw_text)
        metadata_payload["raw_payload_sha256"] = hashlib.sha256(persisted_raw_text.encode("utf-8")).hexdigest() if persisted_raw_text else ""
        metadata_payload["normalization_attempted"] = bool(normalized_payload.get("normalization_attempted"))
        metadata_payload["normalization_applied"] = bool(normalized_payload.get("normalization_applied"))
        metadata_payload["normalization_reason"] = str(normalized_payload.get("normalization_reason", ""))
        metadata_payload["original_wrapper_preserved"] = bool(original_wrapper_text and raw_response_wrapper_path.exists())
        metadata_payload["original_wrapper_path"] = to_repo_rel(raw_response_wrapper_path) if raw_response_wrapper_path.exists() else ""
        metadata_payload["original_wrapper_character_count"] = len(original_wrapper_text)
        metadata_payload["original_wrapper_sha256"] = (
            hashlib.sha256(original_wrapper_text.encode("utf-8")).hexdigest() if original_wrapper_text else ""
        )
        metadata_payload["status"] = str(call_result.get("status", "request_failure"))
        metadata_payload["request_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_payload["response_character_count"] = len(persisted_raw_text)
        metadata_payload["response_sha256"] = metadata_payload["raw_payload_sha256"]
        metadata_payload["elapsed_seconds"] = float(call_result.get("elapsed_seconds") or 0.0)
        metadata_payload["response_status_code"] = int(call_result.get("response_status_code") or 0)
        if metadata_payload["status"] == "success":
            metadata_payload["raw_payload_status"] = "success_payload_persisted" if metadata_payload["raw_payload_persisted"] else "success_without_payload"
            metadata_payload["api_failure"] = ""
            print(f"{progress_label} wrote_raw_response={to_repo_rel(raw_response_path)}", flush=True)
        else:
            metadata_payload["raw_payload_status"] = (
                "request_failure_payload_persisted" if metadata_payload["raw_payload_persisted"] else "request_failure_no_payload"
            )
            metadata_payload["api_failure"] = {
                "error_type": str(call_result.get("error_type", "")),
                "error_message": str(call_result.get("error_message", "")),
            }
            if metadata_payload["raw_payload_persisted"]:
                print(f"{progress_label} wrote_partial_raw_response={to_repo_rel(raw_response_path)}", flush=True)
            print(f"{progress_label} request_failure={call_result.get('error_type', '')}: {call_result.get('error_message', '')}", flush=True)
    except Exception as exc:
        metadata_payload["status"] = "request_failure"
        metadata_payload["request_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_payload["response_character_count"] = len(raw_text)
        metadata_payload["response_sha256"] = hashlib.sha256(raw_text.encode("utf-8")).hexdigest() if raw_text else ""
        metadata_payload["elapsed_seconds"] = 0.0
        metadata_payload["response_status_code"] = 0
        metadata_payload["raw_payload_persisted"] = bool(raw_text and raw_response_path.exists())
        metadata_payload["raw_payload_character_count"] = len(raw_text)
        metadata_payload["raw_payload_sha256"] = metadata_payload["response_sha256"]
        metadata_payload["raw_payload_status"] = "exception_after_payload_persisted" if metadata_payload["raw_payload_persisted"] else "exception_before_payload_persisted"
        metadata_payload["api_failure"] = {"error_type": type(exc).__name__, "error_message": str(exc)}
        print(f"{progress_label} request_failure={type(exc).__name__}: {exc}", flush=True)

    with metadata_write_lock:
        request_metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "paper_key": paper_key,
        "doi": doi,
        "status": metadata_payload["status"],
        "llm_backend": "ollama_local_diagnostic",
        "base_url": base_url,
        "model": model,
        "request_mode": "non_streaming",
        "source_prompts_jsonl_path": to_repo_rel(prompts_jsonl),
        "source_prompt_sha256": prompt_sha256,
        "raw_response_path": to_repo_rel(raw_response_path) if raw_response_path.exists() else "",
        "raw_payload_status": metadata_payload.get("raw_payload_status", ""),
        "raw_payload_persisted": "yes" if metadata_payload.get("raw_payload_persisted") else "no",
        "request_metadata_path": to_repo_rel(request_metadata_path),
        "response_status_code": metadata_payload.get("response_status_code", 0),
        "normalization_attempted": "yes" if metadata_payload.get("normalization_attempted") else "no",
        "normalization_applied": "yes" if metadata_payload.get("normalization_applied") else "no",
        "normalization_reason": metadata_payload.get("normalization_reason", ""),
        "api_failure_type": metadata_payload["api_failure"].get("error_type", "") if isinstance(metadata_payload["api_failure"], dict) else "",
        "api_failure_message": metadata_payload["api_failure"].get("error_message", "") if isinstance(metadata_payload["api_failure"], dict) else "",
    }


def build_run_context(
    *,
    run_id: str,
    run_dir: Path,
    run_dir_kind: str,
    run_selection_mode: str,
    bucket_dir: Path,
    prompts_jsonl: Path,
    prompt_template: Path | None,
    prompt_audit: Path | None,
    freeze_manifest: Path | None,
    selected_paper_keys: list[str],
    base_url: str,
    model: str,
    request_timeout_seconds: int,
    request_retries: int,
    retry_sleep_sec: float,
    max_parallel_requests: int,
    inter_request_sleep_seconds: float,
    raw_dir: Path,
    metadata_dir: Path,
    summary_path: Path,
    success_count: int,
    failure_count: int,
) -> str:
    key_block = "\n".join(f"- `{key}`" for key in selected_paper_keys) if selected_paper_keys else "- `all prompt rows`"
    return f"""# RUN_CONTEXT

## 1. Run ID
`{run_id}`

## 1a. Run Path
- run_dir: `{run_dir}`
- run_dir_kind: `{run_dir_kind}`
- run_selection_mode: `{run_selection_mode}`
- bucket_dir: `{bucket_dir}`

## 2. Run Type
`intermediate_diagnostic_run`

Benchmark reporting rule:
- This run is `diagnostic-only, local-only, non-mainline, not benchmark-valid final output`.
- It executes an `S2-4b-style diagnostic boundary` only.
- It does not alter the maintained Gemini S2-4b path or any downstream Stage2 completion semantics.
- It does not create a lawful Stage3 resume boundary by itself.

## 3. Purpose
- Consume frozen immutable `S2-4a` prompt artifacts only.
- Execute a local Ollama call path adjacent to the maintained `S2-4b` boundary for diagnostic use only.
- Persist raw returned payloads exactly as returned plus request-level metadata sidecars.
- Stop immediately after raw-response persistence without semantic-content judgment.

## 4. Stage Boundary
- current_stage_boundary: `S2-4b-style diagnostic boundary`
- upstream_frozen_dependency: `S2-4a`
- stage_local_owner_script: `src/stage2_sampling_labels/run_stage2_s2_4b_ollama_local_diagnostic_v1.py`
- stage_local_owner_function_surface:
  - `src/stage2_sampling_labels/ollama_local_backend_v1.py::call_ollama_chat_nonstream`
- stop_boundary: `raw_response_payloads_written`
- next_lawful_step: `none inside this diagnostic runner`

## 5. Frozen Inputs
- prompts_jsonl: `{prompts_jsonl}`
- prompt_template: `{prompt_template or ''}`
- prompt_audit: `{prompt_audit or ''}`
- freeze_manifest: `{freeze_manifest or ''}`
- selected_paper_keys:
{key_block}

## 6. Live Call Settings
- llm_backend: `ollama_local_diagnostic`
- base_url: `{base_url}`
- model: `{model}`
- generation_config:
  - `temperature=0`
- request_mode: `non_streaming`
- request_timeout_seconds: `{request_timeout_seconds}`
- request_retries: `{request_retries}`
- retry_sleep_sec: `{retry_sleep_sec}`
- max_parallel_requests: `{max_parallel_requests}`
- inter_request_sleep_seconds: `{inter_request_sleep_seconds}`

## 7. Exact Script Execution Order
1. `src/stage2_sampling_labels/run_stage2_s2_4b_ollama_local_diagnostic_v1.py`
2. `src/stage2_sampling_labels/ollama_local_backend_v1.py::call_ollama_chat_nonstream`

## 8. Outputs
- raw responses:
  - `{raw_dir}`
- per-request metadata sidecars:
  - `{metadata_dir}`
- request summary:
  - `{summary_path}`
- run context:
  - `{run_dir / 'RUN_CONTEXT.md'}`
- machine-readable run metadata:
  - `{run_dir / RUN_METADATA_NAME}`

## 9. Stop Rule
- This run stops after local diagnostic S2-4b-style raw-response persistence.
- The following substeps were not executed:
  - `S2-5 semantic parsing`
  - `S2-6 contract validation`
  - `S2-7 compatibility projection`
  - `Stage3`
  - `Stage4`
  - `Stage5`

## 10. Run Summary
- request_count: `{success_count + failure_count}`
- success_count: `{success_count}`
- failure_count: `{failure_count}`
- success_status: `{"pass" if failure_count == 0 else "partial_fail"}`
"""


def main() -> None:
    args = parse_args()
    base_url = resolve_ollama_base_url(args.base_url)
    model = resolve_ollama_model(args.model)

    prompts_jsonl = repo_path(args.prompts_jsonl)
    prompt_template = optional_repo_path(args.prompt_template)
    prompt_audit = optional_repo_path(args.prompt_audit)
    freeze_manifest = optional_repo_path(args.freeze_manifest)
    if not prompts_jsonl.exists():
        raise FileNotFoundError(f"Frozen prompts JSONL not found: {prompts_jsonl}")
    for optional_path, label in [(prompt_template, "prompt template"), (prompt_audit, "prompt audit"), (freeze_manifest, "freeze manifest")]:
        if optional_path is not None and not optional_path.exists():
            raise FileNotFoundError(f"{label} not found: {optional_path}")

    prompt_rows = filter_prompt_rows(read_jsonl(prompts_jsonl), args.paper_keys)
    if not prompt_rows:
        raise ValueError("No prompt rows were selected for local diagnostic S2-4b-style execution.")
    if args.max_parallel_requests < 1:
        raise ValueError("--max-parallel-requests must be at least 1.")
    if args.inter_request_sleep_seconds < 0:
        raise ValueError("--inter-request-sleep-seconds must be non-negative.")

    target = resolve_results_write_target(
        results_root=DATA_RESULTS_DIR,
        default_child_cue=args.execution_cue,
        explicit_run_dir=repo_path(args.run_dir) if str(args.run_dir).strip() else None,
        explicit_legacy_run_id=args.run_id,
    )
    run_dir = Path(target["run_dir"])
    run_id = target["run_basename"]
    run_dir_kind = target["path_kind"]
    run_selection_mode = target["selection_mode"]
    bucket_dir = Path(target["bucket_dir"])
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    if run_dir_kind == "v2_child_execution":
        bucket_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)

    analysis_dir = run_dir / "analysis"
    raw_dir = run_dir / "raw_responses"
    metadata_dir = run_dir / "request_metadata"
    analysis_dir.mkdir(parents=True, exist_ok=False)
    raw_dir.mkdir(parents=True, exist_ok=False)
    metadata_dir.mkdir(parents=True, exist_ok=False)
    summary_path = analysis_dir / REQUEST_SUMMARY_NAME

    summary_rows: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    total = len(prompt_rows)
    metadata_write_lock = threading.Lock()

    initial_run_context = build_run_context(
        run_id=run_id,
        run_dir=run_dir,
        run_dir_kind=run_dir_kind,
        run_selection_mode=run_selection_mode,
        bucket_dir=bucket_dir,
        prompts_jsonl=prompts_jsonl,
        prompt_template=prompt_template,
        prompt_audit=prompt_audit,
        freeze_manifest=freeze_manifest,
        selected_paper_keys=[str(row.get("paper_key", "")).strip() for row in prompt_rows],
        base_url=base_url,
        model=model,
        request_timeout_seconds=args.request_timeout_seconds,
        request_retries=args.request_retries,
        retry_sleep_sec=args.retry_sleep_sec,
        max_parallel_requests=args.max_parallel_requests,
        inter_request_sleep_seconds=args.inter_request_sleep_seconds,
        raw_dir=raw_dir,
        metadata_dir=metadata_dir,
        summary_path=summary_path,
        success_count=0,
        failure_count=0,
    )
    (run_dir / "RUN_CONTEXT.md").write_text(initial_run_context, encoding="utf-8")

    if args.max_parallel_requests == 1:
        for index, row in enumerate(prompt_rows, start=1):
            result = process_prompt_row(
                index=index,
                total=total,
                row=row,
                base_url=base_url,
                model=model,
                request_timeout_seconds=args.request_timeout_seconds,
                request_retries=args.request_retries,
                retry_sleep_sec=args.retry_sleep_sec,
                prompts_jsonl=prompts_jsonl,
                prompt_template=prompt_template,
                prompt_audit=prompt_audit,
                freeze_manifest=freeze_manifest,
                raw_dir=raw_dir,
                metadata_dir=metadata_dir,
                metadata_write_lock=metadata_write_lock,
            )
            summary_rows.append(result)
            if result["status"] == "success":
                success_count += 1
            else:
                failure_count += 1
            if args.inter_request_sleep_seconds > 0 and index < total:
                print(f"sleeping_between_requests seconds={args.inter_request_sleep_seconds}", flush=True)
                time.sleep(args.inter_request_sleep_seconds)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel_requests) as executor:
            futures: list[concurrent.futures.Future[dict[str, Any]]] = []
            for index, row in enumerate(prompt_rows, start=1):
                futures.append(
                    executor.submit(
                        process_prompt_row,
                        index=index,
                        total=total,
                        row=row,
                        base_url=base_url,
                        model=model,
                        request_timeout_seconds=args.request_timeout_seconds,
                        request_retries=args.request_retries,
                        retry_sleep_sec=args.retry_sleep_sec,
                        prompts_jsonl=prompts_jsonl,
                        prompt_template=prompt_template,
                        prompt_audit=prompt_audit,
                        freeze_manifest=freeze_manifest,
                        raw_dir=raw_dir,
                        metadata_dir=metadata_dir,
                        metadata_write_lock=metadata_write_lock,
                    )
                )
                if args.inter_request_sleep_seconds > 0 and index < total:
                    print(f"sleeping_between_request_submissions seconds={args.inter_request_sleep_seconds}", flush=True)
                    time.sleep(args.inter_request_sleep_seconds)
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                summary_rows.append(result)
                if result["status"] == "success":
                    success_count += 1
                else:
                    failure_count += 1

    write_tsv(
        summary_path,
        [
            "paper_key",
            "doi",
            "status",
            "llm_backend",
            "base_url",
            "model",
            "request_mode",
            "source_prompts_jsonl_path",
            "source_prompt_sha256",
            "raw_response_path",
            "raw_payload_status",
            "raw_payload_persisted",
            "request_metadata_path",
            "response_status_code",
            "normalization_attempted",
            "normalization_applied",
            "normalization_reason",
            "api_failure_type",
            "api_failure_message",
        ],
        summary_rows,
    )

    run_context_text = build_run_context(
        run_id=run_id,
        run_dir=run_dir,
        run_dir_kind=run_dir_kind,
        run_selection_mode=run_selection_mode,
        bucket_dir=bucket_dir,
        prompts_jsonl=prompts_jsonl,
        prompt_template=prompt_template,
        prompt_audit=prompt_audit,
        freeze_manifest=freeze_manifest,
        selected_paper_keys=[str(row.get("paper_key", "")).strip() for row in prompt_rows],
        base_url=base_url,
        model=model,
        request_timeout_seconds=args.request_timeout_seconds,
        request_retries=args.request_retries,
        retry_sleep_sec=args.retry_sleep_sec,
        max_parallel_requests=args.max_parallel_requests,
        inter_request_sleep_seconds=args.inter_request_sleep_seconds,
        raw_dir=raw_dir,
        metadata_dir=metadata_dir,
        summary_path=summary_path,
        success_count=success_count,
        failure_count=failure_count,
    )
    (run_dir / "RUN_CONTEXT.md").write_text(run_context_text, encoding="utf-8")

    run_metadata = {
        "schema": "stage2_s2_4b_ollama_local_diagnostic_run_metadata_v1",
        "run_id": run_id,
        "run_dir": to_repo_rel(run_dir),
        "stage_boundary": "S2-4b_style_diagnostic",
        "upstream_frozen_boundary": "S2-4a",
        "owner_script": "src/stage2_sampling_labels/run_stage2_s2_4b_ollama_local_diagnostic_v1.py",
        "owner_function_surface": ["src/stage2_sampling_labels/ollama_local_backend_v1.py::call_ollama_chat_nonstream"],
        "llm_backend": "ollama_local_diagnostic",
        "base_url": base_url,
        "model": model,
        "request_timeout_seconds": args.request_timeout_seconds,
        "request_retries": args.request_retries,
        "retry_sleep_sec": args.retry_sleep_sec,
        "max_parallel_requests": args.max_parallel_requests,
        "inter_request_sleep_seconds": args.inter_request_sleep_seconds,
        "generation_config": {"temperature": 0},
        "request_mode": "non_streaming",
        "input_artifacts": {
            "prompts_jsonl": to_repo_rel(prompts_jsonl),
            "prompt_template": to_repo_rel(prompt_template),
            "prompt_audit": to_repo_rel(prompt_audit),
            "freeze_manifest": to_repo_rel(freeze_manifest),
        },
        "selected_paper_keys": [str(row.get("paper_key", "")).strip() for row in prompt_rows],
        "outputs": {
            "raw_responses_dir": to_repo_rel(raw_dir),
            "request_metadata_dir": to_repo_rel(metadata_dir),
            "request_summary_tsv": to_repo_rel(summary_path),
            "run_context": to_repo_rel(run_dir / "RUN_CONTEXT.md"),
        },
        "request_count": total,
        "success_count": success_count,
        "failure_count": failure_count,
        "stop_boundary": "raw_response_payloads_written",
        "not_executed": ["S2-5 semantic parsing", "S2-6 contract validation", "S2-7 compatibility projection", "Stage3", "Stage4", "Stage5"],
        "diagnostic_only_note": "This local Ollama path is diagnostic-only, non-mainline, and not benchmark-valid final output.",
    }
    (run_dir / RUN_METADATA_NAME).write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote_raw_responses_dir={raw_dir}")
    print(f"wrote_request_metadata_dir={metadata_dir}")
    print(f"wrote_request_summary={summary_path}")
    print(f"success_count={success_count}")
    print(f"failure_count={failure_count}")


if __name__ == "__main__":
    main()
