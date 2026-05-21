#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

try:
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import call_gemini_stream_collect
    from src.utils.model_policy import validate_models_or_raise
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import resolve_results_write_target
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import call_gemini_stream_collect
    from src.utils.model_policy import validate_models_or_raise
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import resolve_results_write_target


RUN_METADATA_NAME = "stage2_s2_4b_run_metadata_v1.json"
REQUEST_SUMMARY_NAME = "s2_4b_request_summary_v1.tsv"
RAW_RESPONSE_FILENAME_TEMPLATE = "{paper_key}__stage2_v2_raw_response.json"
REQUEST_METADATA_FILENAME_TEMPLATE = "{paper_key}__stage2_v2_request_metadata.json"
ACTIVE_PARAMETER_LOCK_TOKEN = "active_params"
ACTIVE_PARAMETER_LOCK_FIELDS = (
    "llm_backend",
    "model",
    "deepseek_response_format",
    "deepseek_thinking",
    "deepseek_streaming",
    "max_tokens",
    "request_timeout_seconds",
    "request_retries",
    "retry_sleep_sec",
    "max_parallel_requests",
    "inter_request_sleep_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the frozen S2-4b live LLM call boundary from immutable S2-4a prompt artifacts and explicit backend/model parameters, then stop after raw-response persistence."
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Explicit legacy compatibility run_id. New writes default to MDEC084 v2 bucket/child naming when omitted.",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Explicit results run directory. Supports legacy roots and v2 child execution paths under data/results/.",
    )
    parser.add_argument(
        "--execution-cue",
        default="s2_4b_live_llm_call",
        help="Future-facing child cue used only when auto-allocating a new v2 child execution path.",
    )
    parser.add_argument(
        "--prompts-jsonl",
        required=True,
        help="Canonical frozen S2-4a prompt JSONL artifact. Each row must contain paper_key and prompt_text.",
    )
    parser.add_argument(
        "--prompt-template",
        default="",
        help="Optional frozen S2-4a prompt template path recorded for traceability.",
    )
    parser.add_argument(
        "--prompt-audit",
        default="",
        help="Optional frozen S2-4a prompt audit TSV path recorded for traceability.",
    )
    parser.add_argument(
        "--freeze-manifest",
        default="",
        help="Optional FREEZE_MANIFEST.md path recorded for traceability.",
    )
    parser.add_argument(
        "--paper-key",
        action="append",
        dest="paper_keys",
        default=[],
        help="Repeatable paper key filter. Default: all prompt rows in --prompts-jsonl.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["gemini", "deepseek"],
        default="gemini",
        help="Explicit S2-4b live-call backend. Default preserves the existing Gemini path; DeepSeek uses the OpenAI-compatible chat/completions API.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name for this explicit S2-4b live-call boundary. No repository-wide model default is applied.",
    )
    parser.add_argument(
        "--deepseek-base-url",
        default="https://api.deepseek.com",
        help="DeepSeek OpenAI-compatible base URL. Used only with --llm-backend deepseek.",
    )
    parser.add_argument(
        "--deepseek-thinking",
        choices=["enabled", "disabled"],
        default="disabled",
        help="DeepSeek thinking mode for the S2-4b request. Initial trial default: disabled.",
    )
    parser.add_argument(
        "--deepseek-reasoning-effort",
        choices=["high", "max"],
        default="high",
        help="DeepSeek reasoning effort if thinking is enabled. Recorded for metadata; ignored when thinking is disabled.",
    )
    parser.add_argument(
        "--deepseek-response-format",
        choices=["json_object", "none"],
        default="json_object",
        help="DeepSeek response_format mode. Initial trial default: json_object.",
    )
    parser.add_argument(
        "--deepseek-streaming",
        choices=["enabled", "disabled"],
        default="disabled",
        help="Use DeepSeek chat/completions Server-Sent Events streaming and collect the final text. Default preserves prior non-streaming behavior.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum output tokens for backends that expose max_tokens. Used by DeepSeek; default avoids truncating compact Stage2 JSON.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=180,
        help="Per-request Gemini SDK timeout in seconds for the frozen S2-4b live-call boundary. Current frozen cycle default: 180.",
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=0,
        help="Request-level retries for the frozen S2-4b live-call boundary. Current frozen cycle default: 0.",
    )
    parser.add_argument("--retry-sleep-sec", type=float, default=3.0)
    parser.add_argument(
        "--max-parallel-requests",
        type=int,
        default=1,
        help="Maximum concurrent live requests. Default preserves current serial behavior.",
    )
    parser.add_argument(
        "--inter-request-sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between request submissions. Default preserves current no-pacing behavior.",
    )
    parser.add_argument(
        "--allow-active-parameter-deviation",
        action="store_true",
        help="Permit an explicit, documented deviation from a campaign-local S2-4b active parameter lock.",
    )
    parser.add_argument(
        "--active-parameter-deviation-reason",
        default="",
        help="Required reason when --allow-active-parameter-deviation is used.",
    )
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


def load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('\"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def call_deepseek_chat_completion(
    *,
    model: str,
    prompt_text: str,
    base_url: str,
    thinking: str,
    reasoning_effort: str,
    response_format: str,
    max_tokens: int,
    timeout_seconds: int,
    streaming: str = "disabled",
) -> dict[str, Any]:
    load_env_file()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing in environment.")
    started = time.monotonic()
    url = base_url.rstrip("/") + "/chat/completions"
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": streaming == "enabled",
        "max_tokens": max(1, max_tokens),
        "thinking": {"type": thinking},
    }
    if streaming == "enabled":
        body["stream_options"] = {"include_usage": True}
    if thinking == "enabled":
        body["reasoning_effort"] = reasoning_effort
    if response_format == "json_object":
        body["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if streaming == "enabled" else "application/json",
        },
        method="POST",
    )
    if streaming == "enabled":
        chunks: list[str] = []
        reasoning_chunks: list[str] = []
        usage: dict[str, Any] = {}
        chunk_count = 0
        first_chunk_elapsed_seconds = 0.0
        try:
            with urllib.request.urlopen(request, timeout=max(1, timeout_seconds)) as response:
                status_code = int(getattr(response, "status", 0) or 0)
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line.split("data:", 1)[1].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if first_chunk_elapsed_seconds <= 0:
                        first_chunk_elapsed_seconds = time.monotonic() - started
                    chunk_count += 1
                    if isinstance(event, dict) and isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    choices = event.get("choices") if isinstance(event, dict) else None
                    if not isinstance(choices, list) or not choices:
                        continue
                    delta = choices[0].get("delta", {}) if isinstance(choices[0], dict) else {}
                    if not isinstance(delta, dict):
                        continue
                    content_piece = delta.get("content")
                    reasoning_piece = delta.get("reasoning_content")
                    if content_piece:
                        chunks.append(str(content_piece))
                    if reasoning_piece:
                        reasoning_chunks.append(str(reasoning_piece))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            return {
                "status": "request_failure",
                "error_type": "HTTPError",
                "error_message": f"HTTP {exc.code}: {error_body[:1000]}",
                "http_status": exc.code,
                "error_body": error_body,
                "elapsed_seconds": time.monotonic() - started,
                "chunk_count": chunk_count,
                "first_chunk_elapsed_seconds": first_chunk_elapsed_seconds,
            }
        except Exception as exc:
            return {
                "status": "request_failure",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "elapsed_seconds": time.monotonic() - started,
                "chunk_count": chunk_count,
                "first_chunk_elapsed_seconds": first_chunk_elapsed_seconds,
            }
        content = "".join(chunks)
        reasoning_content = "".join(reasoning_chunks)
        if not content.strip():
            return {
                "status": "empty_content",
                "text": "",
                "http_status": status_code,
                "reasoning_content": reasoning_content,
                "usage": usage,
                "elapsed_seconds": time.monotonic() - started,
                "chunk_count": chunk_count,
                "first_chunk_elapsed_seconds": first_chunk_elapsed_seconds,
            }
        return {
            "status": "success",
            "text": content,
            "http_status": status_code,
            "reasoning_content": reasoning_content,
            "usage": usage,
            "elapsed_seconds": time.monotonic() - started,
            "chunk_count": chunk_count,
            "first_chunk_elapsed_seconds": first_chunk_elapsed_seconds,
        }
    try:
        with urllib.request.urlopen(request, timeout=max(1, timeout_seconds)) as response:
            response_text = response.read().decode("utf-8")
            status_code = int(getattr(response, "status", 0) or 0)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        return {
            "status": "request_failure",
            "error_type": "HTTPError",
            "error_message": f"HTTP {exc.code}: {error_body[:1000]}",
            "http_status": exc.code,
            "error_body": error_body,
            "elapsed_seconds": time.monotonic() - started,
        }
    except Exception as exc:
        return {
            "status": "request_failure",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "elapsed_seconds": time.monotonic() - started,
        }
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        return {
            "status": "request_failure",
            "error_type": "JSONDecodeError",
            "error_message": str(exc),
            "http_status": status_code,
            "response_text": response_text,
            "elapsed_seconds": time.monotonic() - started,
        }
    choices = payload.get("choices") if isinstance(payload, dict) else None
    message = choices[0].get("message", {}) if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    content = str(message.get("content") or "") if isinstance(message, dict) else ""
    reasoning_content = str(message.get("reasoning_content") or "") if isinstance(message, dict) else ""
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    if not content.strip():
        return {
            "status": "empty_content",
            "text": "",
            "http_status": status_code,
            "api_payload": payload,
            "reasoning_content": reasoning_content,
            "usage": usage if isinstance(usage, dict) else {},
            "elapsed_seconds": time.monotonic() - started,
        }
    return {
        "status": "success",
        "text": content,
        "http_status": status_code,
        "api_payload": payload,
        "reasoning_content": reasoning_content,
        "usage": usage if isinstance(usage, dict) else {},
        "elapsed_seconds": time.monotonic() - started,
    }


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


def path_campaign_bucket(path: Path) -> Path | None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(DATA_RESULTS_DIR.resolve())
    except ValueError:
        return None
    parts = relative.parts
    if not parts:
        return None
    return DATA_RESULTS_DIR.resolve() / parts[0]


def parse_run_context_active_parameters(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    parsed: dict[str, Any] = {"lock_source": str(path)}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("- llm_backend:"):
            parsed["llm_backend"] = line.split("`", 2)[1] if "`" in line else line.split(":", 1)[1].strip()
        elif line.startswith("- model:"):
            parsed["model"] = line.split("`", 2)[1] if "`" in line else line.split(":", 1)[1].strip()
        elif "deepseek:" in line and "response_format=" in line:
            for token, field in [
                ("response_format=", "deepseek_response_format"),
                ("thinking.type=", "deepseek_thinking"),
                ("streaming=", "deepseek_streaming"),
                ("max_tokens=", "max_tokens"),
            ]:
                if token in line:
                    value = line.split(token, 1)[1].split("`", 1)[0].split(",", 1)[0].strip()
                    parsed[field] = int(value) if field == "max_tokens" and value.isdigit() else value
        elif line.startswith("- request_timeout_seconds:"):
            value = line.split("`", 2)[1] if "`" in line else line.split(":", 1)[1].strip()
            parsed["request_timeout_seconds"] = int(value)
        elif line.startswith("- request_retries:"):
            value = line.split("`", 2)[1] if "`" in line else line.split(":", 1)[1].strip()
            parsed["request_retries"] = int(value)
        elif line.startswith("- retry_sleep_sec:"):
            value = line.split("`", 2)[1] if "`" in line else line.split(":", 1)[1].strip()
            parsed["retry_sleep_sec"] = float(value)
        elif line.startswith("- max_parallel_requests:"):
            value = line.split("`", 2)[1] if "`" in line else line.split(":", 1)[1].strip()
            parsed["max_parallel_requests"] = int(value)
        elif line.startswith("- inter_request_sleep_seconds:"):
            value = line.split("`", 2)[1] if "`" in line else line.split(":", 1)[1].strip()
            parsed["inter_request_sleep_seconds"] = float(value)
    return parsed


def discover_campaign_active_parameter_lock(prompts_jsonl: Path, run_dir: Path) -> dict[str, Any] | None:
    prompt_bucket = path_campaign_bucket(prompts_jsonl)
    run_bucket = path_campaign_bucket(run_dir)
    bucket = prompt_bucket if prompt_bucket is not None else run_bucket
    if bucket is None or not bucket.exists():
        return None
    candidates: list[Path] = []
    for child in sorted(bucket.iterdir()):
        if not child.is_dir():
            continue
        name = child.name.lower()
        if "s2_4b" not in name or ACTIVE_PARAMETER_LOCK_TOKEN not in name:
            continue
        context_path = child / "RUN_CONTEXT.md"
        if context_path.exists():
            candidates.append(context_path)
    if not candidates:
        return None
    parsed_locks = [parse_run_context_active_parameters(candidate) for candidate in candidates]
    first_signature = {field: parsed_locks[0].get(field) for field in ACTIVE_PARAMETER_LOCK_FIELDS}
    conflicting_locks = [
        lock
        for lock in parsed_locks[1:]
        if {field: lock.get(field) for field in ACTIVE_PARAMETER_LOCK_FIELDS} != first_signature
    ]
    if conflicting_locks:
        candidate_list = ", ".join(to_repo_rel(candidate) for candidate in candidates)
        raise ValueError(
            "Multiple campaign-local S2-4b active parameter locks were found. "
            f"Refuse ambiguous live-call execution before model use: {candidate_list}"
        )
    lock = parsed_locks[0]
    lock["campaign_bucket"] = to_repo_rel(bucket)
    lock["lock_sources"] = ";".join(to_repo_rel(candidate) for candidate in candidates)
    return lock


def validate_against_active_parameter_lock(args: argparse.Namespace, lock: dict[str, Any] | None) -> None:
    if not lock:
        return
    actual = {
        "llm_backend": args.llm_backend,
        "model": args.model,
        "deepseek_response_format": args.deepseek_response_format,
        "deepseek_thinking": args.deepseek_thinking,
        "deepseek_streaming": args.deepseek_streaming,
        "max_tokens": args.max_tokens,
        "request_timeout_seconds": args.request_timeout_seconds,
        "request_retries": args.request_retries,
        "retry_sleep_sec": float(args.retry_sleep_sec),
        "max_parallel_requests": args.max_parallel_requests,
        "inter_request_sleep_seconds": float(args.inter_request_sleep_seconds),
    }
    mismatches: list[str] = []
    for field, actual_value in actual.items():
        if field not in lock:
            continue
        expected_value = lock[field]
        if isinstance(expected_value, float):
            matched = abs(float(actual_value) - expected_value) < 1e-9
        else:
            matched = actual_value == expected_value
        if not matched:
            mismatches.append(f"{field}: expected={expected_value!r} actual={actual_value!r}")
    if not mismatches:
        return
    if args.allow_active_parameter_deviation:
        if not str(args.active_parameter_deviation_reason).strip():
            raise ValueError(
                "--allow-active-parameter-deviation requires --active-parameter-deviation-reason before S2-4b live calls."
            )
        return
    mismatch_text = "; ".join(mismatches)
    raise ValueError(
        "S2-4b active parameter lock mismatch. "
        f"lock_source={lock.get('lock_source', '')}; campaign_bucket={lock.get('campaign_bucket', '')}; {mismatch_text}. "
        "Use the campaign active parameters or provide --allow-active-parameter-deviation with a documented reason."
    )


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
    llm_backend: str,
    model: str,
    request_timeout_seconds: int,
    request_retries: int,
    retry_sleep_sec: float,
    deepseek_base_url: str,
    deepseek_thinking: str,
    deepseek_reasoning_effort: str,
    deepseek_response_format: str,
    deepseek_streaming: str,
    max_tokens: int,
    prompts_jsonl: Path,
    prompt_template: Path | None,
    prompt_audit: Path | None,
    freeze_manifest: Path | None,
    prompt_sha256: str,
    row: dict[str, Any],
    raw_response_path: Path,
) -> dict[str, Any]:
    return {
        "stage_boundary": "S2-4b",
        "paper_key": paper_key,
        "doi": doi,
        "status": "",
        "llm_backend": llm_backend,
        "model": model,
        "request_timeout_seconds": request_timeout_seconds,
        "request_retries": request_retries,
        "retry_sleep_sec": retry_sleep_sec,
        "generation_config": (
            {"temperature": 0, "response_mime_type": "application/json"}
            if llm_backend == "gemini"
            else {
                "response_format": {"type": "json_object"} if deepseek_response_format == "json_object" else None,
                "thinking": {"type": deepseek_thinking},
                "reasoning_effort": deepseek_reasoning_effort if deepseek_thinking == "enabled" else "",
                "max_tokens": max_tokens,
                "stream": deepseek_streaming == "enabled",
                "stream_options": {"include_usage": True} if deepseek_streaming == "enabled" else None,
            }
        ),
        "deepseek_base_url": deepseek_base_url if llm_backend == "deepseek" else "",
        "request_mode": (
            "stream_collect"
            if llm_backend == "gemini"
            else ("streaming_chat_completions" if deepseek_streaming == "enabled" else "non_streaming_chat_completions")
        ),
        "source_prompts_jsonl_path": to_repo_rel(prompts_jsonl),
        "source_prompt_template_path": to_repo_rel(prompt_template),
        "source_prompt_audit_path": to_repo_rel(prompt_audit),
        "source_freeze_manifest_path": to_repo_rel(freeze_manifest),
        "source_prompt_sha256": prompt_sha256,
        "authority_run_dir": str(row.get("authority_run_dir", "")).strip(),
        "authority_payload_root": str(row.get("authority_payload_root", "")).strip(),
        "source_prompt_row_fields": {
            "paper_key": paper_key,
            "doi": doi,
            "title": str(row.get("title", "")).strip(),
            "evidence_blocks_path": str(row.get("evidence_blocks_path", "")).strip(),
            "source_evidence_artifact_path": str(row.get("source_evidence_artifact_path", "")).strip(),
            "authority_run_dir": str(row.get("authority_run_dir", "")).strip(),
            "authority_payload_root": str(row.get("authority_payload_root", "")).strip(),
            "ordered_block_order": row.get("ordered_block_order", []),
        },
        "request_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_response_path": to_repo_rel(raw_response_path),
        "raw_payload_status": "not_written_yet",
        "raw_payload_persisted": False,
        "raw_payload_character_count": 0,
        "raw_payload_sha256": "",
    }


def process_prompt_row(
    *,
    index: int,
    total: int,
    row: dict[str, Any],
    llm_backend: str,
    model: str,
    request_timeout_seconds: int,
    request_retries: int,
    retry_sleep_sec: float,
    deepseek_base_url: str,
    deepseek_thinking: str,
    deepseek_reasoning_effort: str,
    deepseek_response_format: str,
    deepseek_streaming: str,
    max_tokens: int,
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
    request_metadata_path = metadata_dir / REQUEST_METADATA_FILENAME_TEMPLATE.format(paper_key=paper_key)
    progress_label = f"[{index}/{total}] {paper_key}"
    print(
        f"{progress_label} sending prompt source={to_repo_rel(prompts_jsonl)} prompt_sha256={prompt_sha256} backend={llm_backend} model={model}",
        flush=True,
    )

    metadata_payload = build_request_metadata_payload(
        paper_key=paper_key,
        doi=doi,
        llm_backend=llm_backend,
        model=model,
        request_timeout_seconds=request_timeout_seconds,
        request_retries=request_retries,
        retry_sleep_sec=retry_sleep_sec,
        deepseek_base_url=deepseek_base_url,
        deepseek_thinking=deepseek_thinking,
        deepseek_reasoning_effort=deepseek_reasoning_effort,
        deepseek_response_format=deepseek_response_format,
        deepseek_streaming=deepseek_streaming,
        max_tokens=max_tokens,
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
        if llm_backend == "deepseek":
            call_result = call_deepseek_chat_completion(
                model=model,
                prompt_text=prompt_text,
                base_url=deepseek_base_url,
                thinking=deepseek_thinking,
                reasoning_effort=deepseek_reasoning_effort,
                response_format=deepseek_response_format,
                max_tokens=max_tokens,
                timeout_seconds=max(1, request_timeout_seconds),
                streaming=deepseek_streaming,
            )
        else:
            call_result = call_gemini_stream_collect(
                model,
                prompt_text,
                request_retries,
                retry_sleep_sec,
                progress_label=progress_label,
                timeout_seconds=max(1, request_timeout_seconds),
            )
        raw_text = str(call_result.get("text", "") or "")
        if raw_text:
            raw_response_path.write_text(raw_text, encoding="utf-8")
        metadata_payload["raw_payload_persisted"] = bool(raw_text and raw_response_path.exists())
        metadata_payload["raw_payload_character_count"] = len(raw_text)
        metadata_payload["raw_payload_sha256"] = hashlib.sha256(raw_text.encode("utf-8")).hexdigest() if raw_text else ""
        metadata_payload["status"] = str(call_result.get("status", "request_failure"))
        metadata_payload["request_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_payload["response_character_count"] = len(raw_text)
        metadata_payload["response_sha256"] = metadata_payload["raw_payload_sha256"]
        metadata_payload["stream_chunk_count"] = int(call_result.get("chunk_count") or 0)
        metadata_payload["first_chunk_elapsed_seconds"] = float(call_result.get("first_chunk_elapsed_seconds") or 0.0)
        metadata_payload["elapsed_seconds"] = float(call_result.get("elapsed_seconds") or 0.0)
        metadata_payload["http_status"] = call_result.get("http_status", "")
        metadata_payload["usage"] = call_result.get("usage", {}) if isinstance(call_result.get("usage", {}), dict) else {}
        metadata_payload["deepseek_reasoning_content_character_count"] = len(str(call_result.get("reasoning_content", "") or ""))
        metadata_payload["deepseek_prompt_cache_hit_tokens"] = metadata_payload["usage"].get("prompt_cache_hit_tokens", "") if isinstance(metadata_payload["usage"], dict) else ""
        metadata_payload["deepseek_prompt_cache_miss_tokens"] = metadata_payload["usage"].get("prompt_cache_miss_tokens", "") if isinstance(metadata_payload["usage"], dict) else ""
        if metadata_payload["status"] == "success":
            metadata_payload["raw_payload_status"] = (
                "success_payload_persisted" if metadata_payload["raw_payload_persisted"] else "success_without_payload"
            )
            metadata_payload["api_failure"] = ""
            print(f"{progress_label} wrote_raw_response={to_repo_rel(raw_response_path)}", flush=True)
        else:
            metadata_payload["raw_payload_status"] = (
                "request_failure_payload_persisted"
                if metadata_payload["raw_payload_persisted"]
                else "request_failure_no_payload"
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
        metadata_payload["stream_chunk_count"] = 0
        metadata_payload["first_chunk_elapsed_seconds"] = 0.0
        metadata_payload["elapsed_seconds"] = 0.0
        metadata_payload["raw_payload_persisted"] = bool(raw_text and raw_response_path.exists())
        metadata_payload["raw_payload_character_count"] = len(raw_text)
        metadata_payload["raw_payload_sha256"] = metadata_payload["response_sha256"]
        metadata_payload["raw_payload_status"] = (
            "exception_after_payload_persisted"
            if metadata_payload["raw_payload_persisted"]
            else "exception_before_payload_persisted"
        )
        metadata_payload["api_failure"] = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        print(f"{progress_label} request_failure={type(exc).__name__}: {exc}", flush=True)

    with metadata_write_lock:
        request_metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "paper_key": paper_key,
        "doi": doi,
        "status": metadata_payload["status"],
        "model": model,
        "source_prompts_jsonl_path": to_repo_rel(prompts_jsonl),
        "source_prompt_sha256": prompt_sha256,
        "raw_response_path": to_repo_rel(raw_response_path) if raw_response_path.exists() else "",
        "raw_payload_status": metadata_payload.get("raw_payload_status", ""),
        "raw_payload_persisted": "yes" if metadata_payload.get("raw_payload_persisted") else "no",
        "request_metadata_path": to_repo_rel(request_metadata_path),
        "api_failure_type": (
            metadata_payload["api_failure"].get("error_type", "")
            if isinstance(metadata_payload["api_failure"], dict)
            else ""
        ),
        "api_failure_message": (
            metadata_payload["api_failure"].get("error_message", "")
            if isinstance(metadata_payload["api_failure"], dict)
            else ""
        ),
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
    llm_backend: str,
    model: str,
    deepseek_thinking: str,
    deepseek_response_format: str,
    deepseek_streaming: str,
    max_tokens: int,
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
    active_parameter_lock: dict[str, Any] | None,
    allow_active_parameter_deviation: bool,
    active_parameter_deviation_reason: str,
) -> str:
    key_block = "\n".join(f"- `{key}`" for key in selected_paper_keys) if selected_paper_keys else "- `all prompt rows`"
    lock_block = "- campaign_active_parameter_lock: `not_found`\n"
    if active_parameter_lock:
        lock_block = (
            f"- campaign_active_parameter_lock: `{active_parameter_lock.get('lock_source', '')}`\n"
            f"- active_parameter_deviation_allowed: `{str(bool(allow_active_parameter_deviation)).lower()}`\n"
            f"- active_parameter_deviation_reason: `{active_parameter_deviation_reason}`\n"
        )
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
- This run is `diagnostic-only, not benchmark-valid final output`.
- It executes the frozen `S2-4b` live LLM-call boundary only.
- It does not by itself create a lawful Stage3 resume boundary.
- A frozen current live-v2 raw-response set becomes a downstream-usable Stage2 authority surface only after replay through the maintained composite Stage2 path.

## 3. Purpose
- Consume frozen immutable `S2-4a` prompt artifacts.
- Perform the explicit live LLM call only.
- Persist raw returned payloads and request-level metadata sidecars.
- Stop immediately after raw-response persistence without semantic-content judgment.

## 4. Stage Boundary
- current_stage_boundary: `S2-4b`
- upstream_frozen_dependency: `S2-4a`
- stage_local_owner_script: `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`
- stage_local_owner_function_surface:
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::call_gemini_stream_collect` for Gemini
  - `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py::call_deepseek_chat_completion` for DeepSeek
- stop_boundary: `raw_response_payloads_written`
- next_lawful_step: `S2-5 semantic parsing`, but only if later rehydrated through the maintained composite Stage2 path

## 5. Frozen Inputs
- prompts_jsonl: `{prompts_jsonl}`
- prompt_template: `{prompt_template or ''}`
- prompt_audit: `{prompt_audit or ''}`
- freeze_manifest: `{freeze_manifest or ''}`
- selected_paper_keys:
{key_block}

## 6. Live Call Settings
- llm_backend: `{llm_backend}`
- model: `{model}`
- generation_config:
  - gemini: `temperature=0`, `response_mime_type=application/json`
  - deepseek: `response_format={deepseek_response_format}`, `thinking.type={deepseek_thinking}`, `streaming={deepseek_streaming}`, `max_tokens={max_tokens}`
- request_timeout_seconds: `{request_timeout_seconds}`
- request_retries: `{request_retries}`
- retry_sleep_sec: `{retry_sleep_sec}`
- max_parallel_requests: `{max_parallel_requests}`
- inter_request_sleep_seconds: `{inter_request_sleep_seconds}`
- failure_policy:
  - request/API/transport failures mark failure
  - returned malformed or nonconforming content is still persisted and does not mark failure at this boundary

## 7. Exact Script Execution Order
1. `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`
2. backend-specific live call helper selected by explicit `--llm-backend`

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
- This run stops after S2-4b raw-response persistence.
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

## 11. Active Parameter Lock
{lock_block}"""


def main() -> None:
    args = parse_args()
    validate_models_or_raise([args.model], context="S2-4b live-call runner")

    prompts_jsonl = repo_path(args.prompts_jsonl)
    prompt_template = optional_repo_path(args.prompt_template)
    prompt_audit = optional_repo_path(args.prompt_audit)
    freeze_manifest = optional_repo_path(args.freeze_manifest)
    if not prompts_jsonl.exists():
        raise FileNotFoundError(f"Frozen prompts JSONL not found: {prompts_jsonl}")
    for optional_path, label in [
        (prompt_template, "prompt template"),
        (prompt_audit, "prompt audit"),
        (freeze_manifest, "freeze manifest"),
    ]:
        if optional_path is not None and not optional_path.exists():
            raise FileNotFoundError(f"{label} not found: {optional_path}")

    prompt_rows = filter_prompt_rows(read_jsonl(prompts_jsonl), args.paper_keys)
    if not prompt_rows:
        raise ValueError("No prompt rows were selected for S2-4b execution.")
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
    active_parameter_lock = discover_campaign_active_parameter_lock(prompts_jsonl, run_dir)
    validate_against_active_parameter_lock(args, active_parameter_lock)
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
        llm_backend=args.llm_backend,
        model=args.model,
        deepseek_thinking=args.deepseek_thinking,
        deepseek_response_format=args.deepseek_response_format,
        deepseek_streaming=args.deepseek_streaming,
        max_tokens=args.max_tokens,
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
        active_parameter_lock=active_parameter_lock,
        allow_active_parameter_deviation=args.allow_active_parameter_deviation,
        active_parameter_deviation_reason=args.active_parameter_deviation_reason,
    )
    (run_dir / "RUN_CONTEXT.md").write_text(initial_run_context, encoding="utf-8")

    if args.max_parallel_requests == 1:
        for index, row in enumerate(prompt_rows, start=1):
            result = process_prompt_row(
                index=index,
                total=total,
                row=row,
                llm_backend=args.llm_backend,
                model=args.model,
                request_timeout_seconds=args.request_timeout_seconds,
                request_retries=args.request_retries,
                retry_sleep_sec=args.retry_sleep_sec,
                deepseek_base_url=args.deepseek_base_url,
                deepseek_thinking=args.deepseek_thinking,
                deepseek_reasoning_effort=args.deepseek_reasoning_effort,
                deepseek_response_format=args.deepseek_response_format,
                deepseek_streaming=args.deepseek_streaming,
                max_tokens=args.max_tokens,
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
                        llm_backend=args.llm_backend,
                        model=args.model,
                        request_timeout_seconds=args.request_timeout_seconds,
                        request_retries=args.request_retries,
                        retry_sleep_sec=args.retry_sleep_sec,
                        deepseek_base_url=args.deepseek_base_url,
                        deepseek_thinking=args.deepseek_thinking,
                        deepseek_reasoning_effort=args.deepseek_reasoning_effort,
                        deepseek_response_format=args.deepseek_response_format,
                        deepseek_streaming=args.deepseek_streaming,
                        max_tokens=args.max_tokens,
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
            "model",
            "source_prompts_jsonl_path",
            "source_prompt_sha256",
            "raw_response_path",
            "raw_payload_status",
            "raw_payload_persisted",
            "request_metadata_path",
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
        llm_backend=args.llm_backend,
        model=args.model,
        deepseek_thinking=args.deepseek_thinking,
        deepseek_response_format=args.deepseek_response_format,
        deepseek_streaming=args.deepseek_streaming,
        max_tokens=args.max_tokens,
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
        active_parameter_lock=active_parameter_lock,
        allow_active_parameter_deviation=args.allow_active_parameter_deviation,
        active_parameter_deviation_reason=args.active_parameter_deviation_reason,
    )
    (run_dir / "RUN_CONTEXT.md").write_text(run_context_text, encoding="utf-8")

    run_metadata = {
        "run_id": run_id,
        "run_dir": to_repo_rel(run_dir),
        "stage_boundary": "S2-4b",
        "upstream_frozen_boundary": "S2-4a",
        "owner_script": "src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py",
        "owner_function_surface": [
            "src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::call_gemini_stream_collect",
        ],
        "llm_backend": args.llm_backend,
        "model": args.model,
        "request_timeout_seconds": args.request_timeout_seconds,
        "request_retries": args.request_retries,
        "retry_sleep_sec": args.retry_sleep_sec,
        "max_parallel_requests": args.max_parallel_requests,
        "inter_request_sleep_seconds": args.inter_request_sleep_seconds,
        "generation_config": (
            {"temperature": 0, "response_mime_type": "application/json"}
            if args.llm_backend == "gemini"
            else {
                "response_format": {"type": "json_object"} if args.deepseek_response_format == "json_object" else None,
                "thinking": {"type": args.deepseek_thinking},
                "reasoning_effort": args.deepseek_reasoning_effort if args.deepseek_thinking == "enabled" else "",
                "max_tokens": args.max_tokens,
                "stream": args.deepseek_streaming == "enabled",
                "stream_options": {"include_usage": True} if args.deepseek_streaming == "enabled" else None,
            }
        ),
        "deepseek_base_url": args.deepseek_base_url if args.llm_backend == "deepseek" else "",
        "request_mode": (
            "stream_collect"
            if args.llm_backend == "gemini"
            else ("streaming_chat_completions" if args.deepseek_streaming == "enabled" else "non_streaming_chat_completions")
        ),
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
        "not_executed": [
            "S2-5 semantic parsing",
            "S2-6 contract validation",
            "S2-7 compatibility projection",
            "Stage3",
            "Stage4",
            "Stage5",
        ],
        "diagnostic_only_note": "This run is diagnostic-only, not benchmark-valid final output. Returned payloads were not judged for semantic correctness or structural compliance at this boundary.",
        "active_parameter_lock": active_parameter_lock or {},
        "active_parameter_deviation_allowed": bool(args.allow_active_parameter_deviation),
        "active_parameter_deviation_reason": str(args.active_parameter_deviation_reason).strip(),
    }
    (run_dir / RUN_METADATA_NAME).write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote_raw_responses_dir={raw_dir}")
    print(f"wrote_request_metadata_dir={metadata_dir}")
    print(f"wrote_request_summary={summary_path}")
    print(f"success_count={success_count}")
    print(f"failure_count={failure_count}")


if __name__ == "__main__":
    main()
