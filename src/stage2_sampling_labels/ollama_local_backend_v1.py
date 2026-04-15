#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urljoin


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def resolve_ollama_base_url(explicit_base_url: str = "") -> str:
    value = str(explicit_base_url or "").strip() or str(os.getenv("OLLAMA_BASE_URL", "")).strip()
    if not value:
        value = DEFAULT_OLLAMA_BASE_URL
    return value.rstrip("/") + "/"


def resolve_ollama_model(explicit_model: str = "") -> str:
    value = str(explicit_model or "").strip() or str(os.getenv("OLLAMA_MODEL", "")).strip()
    if not value:
        raise ValueError("Ollama model is required. Pass --model or set OLLAMA_MODEL.")
    return value


def _build_chat_url(base_url: str) -> str:
    return urljoin(resolve_ollama_base_url(base_url), "api/chat")


def _build_version_url(base_url: str) -> str:
    return urljoin(resolve_ollama_base_url(base_url), "api/version")


def _build_tags_url(base_url: str) -> str:
    return urljoin(resolve_ollama_base_url(base_url), "api/tags")


def call_ollama_chat_nonstream(
    *,
    base_url: str,
    model: str,
    prompt: str,
    timeout_seconds: int,
    retries: int,
    sleep_sec: float,
    progress_label: str = "",
) -> dict[str, Any]:
    chat_url = _build_chat_url(base_url)
    payload = {
        "model": resolve_ollama_model(model),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }
    headers = {"Content-Type": "application/json"}
    last_result: dict[str, Any] | None = None

    for attempt in range(retries + 1):
        started_at = time.time()
        response_text = ""
        status_code = 0
        try:
            if progress_label:
                print(
                    f"{progress_label} ollama_request_start attempt={attempt + 1}/{retries + 1} "
                    f"prompt_chars={len(prompt)} timeout_seconds={timeout_seconds} base_url={resolve_ollama_base_url(base_url)}",
                    flush=True,
                )
            request = urllib_request.Request(
                chat_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib_request.urlopen(request, timeout=max(1, timeout_seconds)) as response:
                status_code = int(response.status)
                response_text = response.read().decode("utf-8", errors="replace")
            return {
                "status": "success",
                "text": response_text,
                "elapsed_seconds": round(time.time() - started_at, 3),
                "error_type": "",
                "error_message": "",
                "response_status_code": status_code,
            }
        except urllib_error.HTTPError as exc:
            status_code = int(exc.code)
            response_text = exc.read().decode("utf-8", errors="replace")
            last_result = {
                "status": "request_failure",
                "text": response_text,
                "elapsed_seconds": round(time.time() - started_at, 3),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "response_status_code": status_code,
            }
            if progress_label:
                print(
                    f"{progress_label} ollama_request_exception attempt={attempt + 1}/{retries + 1} "
                    f"status_code={status_code} collected_chars={len(response_text)} "
                    f"error_type={type(exc).__name__} error={exc}",
                    flush=True,
                )
        except Exception as exc:
            last_result = {
                "status": "request_failure",
                "text": response_text,
                "elapsed_seconds": round(time.time() - started_at, 3),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "response_status_code": status_code,
            }
            if progress_label:
                print(
                    f"{progress_label} ollama_request_exception attempt={attempt + 1}/{retries + 1} "
                    f"status_code={status_code} collected_chars={len(response_text)} "
                    f"error_type={type(exc).__name__} error={exc}",
                    flush=True,
                )
        if attempt < retries:
            if progress_label:
                print(f"{progress_label} retrying_ollama attempt={attempt + 2}/{retries + 1}", flush=True)
            time.sleep(sleep_sec)

    return last_result or {
        "status": "request_failure",
        "text": "",
        "elapsed_seconds": 0.0,
        "error_type": "RuntimeError",
        "error_message": "Ollama request failed.",
        "response_status_code": 0,
    }


def check_ollama_connectivity(
    *,
    base_url: str,
    model: str = "",
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    resolved_base_url = resolve_ollama_base_url(base_url)
    requested_model = str(model or "").strip() or str(os.getenv("OLLAMA_MODEL", "")).strip()
    version_url = _build_version_url(resolved_base_url)
    tags_url = _build_tags_url(resolved_base_url)

    report: dict[str, Any] = {
        "schema": "ollama_connectivity_check_v1",
        "base_url": resolved_base_url,
        "requested_model": requested_model,
        "reachable": False,
        "version_endpoint": {"url": version_url, "ok": False, "status_code": 0, "error": ""},
        "tags_endpoint": {"url": tags_url, "ok": False, "status_code": 0, "error": ""},
        "available_models": [],
        "model_available": None,
        "success": False,
    }

    try:
        version_request = urllib_request.Request(version_url, method="GET")
        with urllib_request.urlopen(version_request, timeout=max(1, timeout_seconds)) as version_response:
            report["version_endpoint"]["status_code"] = int(version_response.status)
            report["version_endpoint"]["ok"] = True
            report["version_endpoint"]["body"] = version_response.read().decode("utf-8", errors="replace")
            report["reachable"] = True
    except urllib_error.HTTPError as exc:
        report["version_endpoint"]["status_code"] = int(exc.code)
        report["version_endpoint"]["error"] = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        report["version_endpoint"]["error"] = f"{type(exc).__name__}: {exc}"

    try:
        tags_request = urllib_request.Request(tags_url, method="GET")
        with urllib_request.urlopen(tags_request, timeout=max(1, timeout_seconds)) as tags_response:
            report["tags_endpoint"]["status_code"] = int(tags_response.status)
            report["tags_endpoint"]["ok"] = True
            report["reachable"] = True
            parsed = json.loads(tags_response.read().decode("utf-8", errors="replace"))
            models = parsed.get("models") if isinstance(parsed, dict) else []
            if isinstance(models, list):
                available = []
                for item in models:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name", "")).strip()
                    if name:
                        available.append(name)
                report["available_models"] = available
                if requested_model:
                    report["model_available"] = requested_model in available
    except urllib_error.HTTPError as exc:
        report["tags_endpoint"]["status_code"] = int(exc.code)
        report["tags_endpoint"]["error"] = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        report["tags_endpoint"]["error"] = f"{type(exc).__name__}: {exc}"

    if requested_model and report["model_available"] is None:
        report["model_available"] = False
    report["success"] = bool(report["reachable"]) and (report["model_available"] in (None, True))
    return report
