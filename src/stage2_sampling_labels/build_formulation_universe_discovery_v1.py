#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

try:
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
        call_gemini,
        call_gemini_stream_collect,
    )
    from src.stage2_sampling_labels.run_stage2_s2_4b_live_llm_call_v1 import (
        call_deepseek_chat_completion,
    )
    from src.utils.paths import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
        call_gemini,
        call_gemini_stream_collect,
    )
    from src.stage2_sampling_labels.run_stage2_s2_4b_live_llm_call_v1 import (
        call_deepseek_chat_completion,
    )
    from src.utils.paths import PROJECT_ROOT


ARTIFACT_VERSION = "formulation_universe_discovery_v1"

INCLUDED_FIELDS = [
    "paper_key",
    "canonical_formulation_id",
    "formulation_label",
    "aliases_json",
    "row_role",
    "identity_basis",
    "preparation_evidence_quote",
    "source_locator",
    "confidence",
    "requires_human_review",
]

EXCLUDED_FIELDS = [
    "paper_key",
    "candidate_id",
    "label",
    "excluded_as",
    "exclusion_reason",
    "evidence_quote",
    "source_locator",
]

REVIEW_FIELDS = [
    "paper_key",
    "candidate_id",
    "label",
    "review_reason",
    "evidence_quote",
    "source_locator",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnostic frozen formulation-universe artifact. The live "
            "LLM may discover row membership, but downstream stages must not "
            "create rows from this artifact without a governed promotion."
        )
    )
    parser.add_argument("--manifest-tsv", action="append", default=[], help="Scope manifest TSV. Repeatable.")
    parser.add_argument(
        "--manifest-current-tsv",
        default="data/cleaned/index/manifest_current.tsv",
        help="Authoritative manifest used to enrich split manifests with current text/table paths.",
    )
    parser.add_argument("--paper-key", action="append", default=[], help="Optional repeatable paper key filter.")
    parser.add_argument("--out-dir", required=True, help="Run-scoped output directory.")
    parser.add_argument("--gt-counts-tsv", default="", help="Optional Layer1 GT counts TSV for diagnostic comparison.")
    parser.add_argument("--llm-backend", choices=["none", "deepseek", "gemini", "ollama"], default="none")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--deepseek-base-url", default="https://api.deepseek.com")
    parser.add_argument("--deepseek-streaming", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-num-ctx", type=int, default=32768)
    parser.add_argument("--ollama-format-json", action="store_true")
    parser.add_argument("--max-source-chars", type=int, default=140000)
    parser.add_argument("--max-table-chars", type=int, default=30000)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout-seconds", type=int, default=240)
    parser.add_argument("--request-retries", type=int, default=0)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--gemini-transport", choices=["stream", "sync"], default="stream")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
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


def normalize_path_text(value: str) -> str:
    return str(value or "").strip().replace("\\", "/")


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
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def first_nonempty(row: dict[str, str], names: list[str]) -> str:
    for name in names:
        value = str(row.get(name, "") or "").strip()
        if value:
            return value
    return ""


def key_of(row: dict[str, str]) -> str:
    return first_nonempty(row, ["paper_key", "key", "paper_id", "zotero_key"])


def merge_scope_rows(manifest_paths: list[Path], current_rows: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    current_asset_fields = {
        "text_path",
        "text_source_type",
        "text_available",
        "table_dir",
        "table_available",
        "structure_path",
        "structure_available",
        "stage1_table_cell_sidecar_path",
        "stage1_table_cell_sidecar_available",
    }
    for manifest_path in manifest_paths:
        for row in load_tsv(manifest_path):
            key = key_of(row)
            if not key:
                continue
            payload = dict(current_rows.get(key, {}))
            for field, value in row.items():
                if not str(value or "").strip():
                    continue
                if field in current_asset_fields and str(payload.get(field, "") or "").strip():
                    continue
                payload[field] = value
            payload["paper_key"] = key
            payload["scope_manifest_tsv"] = to_repo_rel(manifest_path)
            merged[key] = payload
    return [merged[key] for key in sorted(merged)]


def load_current_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    if not path.exists():
        return rows
    for row in load_tsv(path):
        key = key_of(row)
        if key:
            rows[key] = row
    return rows


def read_text_slice(path: Path, max_chars: int) -> tuple[str, bool]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars], True
    return text, False


def collect_table_snippets(table_dir: Path, max_chars: int) -> tuple[str, list[str], bool]:
    if not table_dir.exists() or not table_dir.is_dir() or max_chars <= 0:
        return "", [], False
    chunks: list[str] = []
    paths: list[str] = []
    used = 0
    truncated = False
    for csv_path in sorted(table_dir.glob("*.csv")):
        text = csv_path.read_text(encoding="utf-8", errors="replace")
        chunk = f"\n[TABLE_FILE {csv_path.name}]\n{text}\n"
        if used + len(chunk) > max_chars:
            remaining = max(0, max_chars - used)
            if remaining:
                chunks.append(chunk[:remaining])
                paths.append(to_repo_rel(csv_path))
            truncated = True
            break
        chunks.append(chunk)
        paths.append(to_repo_rel(csv_path))
        used += len(chunk)
    return "".join(chunks), paths, truncated


def build_source_package(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    paper_key = key_of(row)
    text_path_value = normalize_path_text(row.get("text_path", ""))
    if not text_path_value:
        raise FileNotFoundError(f"{paper_key}: text_path is empty")
    text_path = repo_path(text_path_value)
    if not text_path.exists():
        raise FileNotFoundError(f"{paper_key}: text_path does not exist: {text_path}")
    source_text, source_text_truncated = read_text_slice(text_path, args.max_source_chars)
    table_path_value = normalize_path_text(row.get("table_dir", ""))
    table_snippets = ""
    table_files: list[str] = []
    tables_truncated = False
    if table_path_value:
        table_snippets, table_files, tables_truncated = collect_table_snippets(
            repo_path(table_path_value),
            args.max_table_chars,
        )
    return {
        "paper_key": paper_key,
        "title": first_nonempty(row, ["title", "paper_title"]),
        "doi": first_nonempty(row, ["doi", "normalized_doi"]),
        "text_path": to_repo_rel(text_path),
        "text_sha256": stable_hash(source_text),
        "source_text_truncated": source_text_truncated,
        "table_dir": table_path_value,
        "table_files": table_files,
        "tables_truncated": tables_truncated,
        "source_text": source_text,
        "table_snippets": table_snippets,
    }


def build_prompt(package: dict[str, Any]) -> str:
    schema = {
        "paper_key": package["paper_key"],
        "included_formulations": [
            {
                "canonical_formulation_id": "F001",
                "formulation_label": "article-native label",
                "aliases": ["optional aliases"],
                "row_role": "experimental_formulation|blank_control|internal_reference|process_control|other_prepared_instance",
                "identity_basis": "why this is one independent prepared formulation instance",
                "preparation_evidence_quote": "short exact quote from the supplied source",
                "source_locator": "section/table/file/page if available",
                "confidence": "high|medium|low",
                "requires_human_review": False,
            }
        ],
        "excluded_candidates": [
            {
                "candidate_id": "X001",
                "label": "candidate label",
                "excluded_as": "assay_only_derivative|freeze_dried_or_rehydrated_variant|post_treatment_state|design_space_combination_without_prepared_instance|commercial_comparator|method_condition_only|duplicate_alias|not_a_prepared_formulation",
                "exclusion_reason": "why it must not create a row",
                "evidence_quote": "short exact quote",
                "source_locator": "section/table/file/page if available",
            }
        ],
        "unresolved_candidate_review": [
            {
                "candidate_id": "R001",
                "label": "candidate label",
                "review_reason": "why model cannot decide safely",
                "evidence_quote": "short exact quote",
                "source_locator": "section/table/file/page if available",
            }
        ],
        "audit_notes": ["brief notes"],
    }
    return f"""You are performing Stage2 formulation-universe discovery for a PLGA literature extraction system.

Task boundary:
- Decide the full prepared formulation row universe for this one paper.
- Do not extract numeric outcome values except when needed to identify row identity.
- Row creation authority belongs only to this task. Later value-binding stages must not create rows.
- Include every real article-prepared PLGA/PLA/PCL nanoparticle or nanosuspension formulation instance.
- If an experimental-design, optimization, DOE, Box-Behnken, factorial, screening, or formulation table reports rows such as F1-F26, N1-N9, Batch 1-12, or similar article-native formulation/run labels, enumerate every reported run as a separate formulation row when the paper presents them as prepared/evaluated formulation instances. Do not collapse the design table into only optimized or selected rows.
- Include blank/control/internal-reference nanoparticle formulations only when they are article-prepared formulation instances in the same formulation family.
- Exclude assay-only derivatives, post-treatment states, freeze-dried or rehydrated states, release-test conditions, commercial comparators, free-drug solutions, calibration standards, process-control solutions, and design-space combinations unless the paper explicitly reports them as independent prepared nanoparticle formulation instances.
- Freeze-dried/rehydrated measurements should usually be aliases or linked lower-level states of the parent formulation, not new rows.
- Merge aliases for the same formulation and explain why they are the same.
- Every included row and excluded candidate must cite a short exact evidence quote from the supplied source.

Return one JSON object only, conforming to this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Paper metadata:
paper_key: {package['paper_key']}
title: {package.get('title', '')}
doi: {package.get('doi', '')}
text_path: {package.get('text_path', '')}
source_text_truncated: {package.get('source_text_truncated')}
tables_truncated: {package.get('tables_truncated')}

Table snippets, if available:
{package.get('table_snippets', '')}

Full/source text:
{package.get('source_text', '')}
"""


def extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON root is not an object")
    return payload


def normalize_included(paper_key: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("included_formulations") or [], start=1):
        if not isinstance(item, dict):
            continue
        fid = str(item.get("canonical_formulation_id") or f"F{index:03d}").strip()
        rows.append(
            {
                "paper_key": paper_key,
                "canonical_formulation_id": fid,
                "formulation_label": item.get("formulation_label", ""),
                "aliases_json": item.get("aliases", []),
                "row_role": item.get("row_role", ""),
                "identity_basis": item.get("identity_basis", ""),
                "preparation_evidence_quote": item.get("preparation_evidence_quote", ""),
                "source_locator": item.get("source_locator", ""),
                "confidence": item.get("confidence", ""),
                "requires_human_review": item.get("requires_human_review", ""),
            }
        )
    return rows


def normalize_excluded(paper_key: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("excluded_candidates") or [], start=1):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "paper_key": paper_key,
                "candidate_id": item.get("candidate_id") or f"X{index:03d}",
                "label": item.get("label", ""),
                "excluded_as": item.get("excluded_as", ""),
                "exclusion_reason": item.get("exclusion_reason", ""),
                "evidence_quote": item.get("evidence_quote", ""),
                "source_locator": item.get("source_locator", ""),
            }
        )
    return rows


def normalize_review(paper_key: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("unresolved_candidate_review") or [], start=1):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "paper_key": paper_key,
                "candidate_id": item.get("candidate_id") or f"R{index:03d}",
                "label": item.get("label", ""),
                "review_reason": item.get("review_reason", ""),
                "evidence_quote": item.get("evidence_quote", ""),
                "source_locator": item.get("source_locator", ""),
            }
        )
    return rows


def call_llm(args: argparse.Namespace, prompt_text: str) -> dict[str, Any]:
    if args.llm_backend == "none":
        return {
            "text": "",
            "usage": {},
            "request_transport": "none",
        }
    if args.llm_backend == "deepseek":
        return call_deepseek_chat_completion(
            model=args.model,
            prompt_text=prompt_text,
            base_url=args.deepseek_base_url,
            thinking="disabled",
            reasoning_effort="high",
            response_format="json_object",
            max_tokens=args.max_output_tokens,
            timeout_seconds=args.request_timeout_seconds,
            streaming=args.deepseek_streaming,
        )
    if args.llm_backend == "gemini":
        if args.gemini_transport == "sync":
            text = call_gemini(
                args.model,
                prompt_text,
                retries=max(0, args.request_retries),
                sleep_sec=max(0.0, args.retry_sleep_seconds),
                progress_label="",
                timeout_seconds=args.request_timeout_seconds,
            )
            return {
                "status": "success" if text.strip() else "empty_content",
                "text": text,
                "usage": {},
                "request_transport": "sync",
                "call_result": {"status": "success" if text.strip() else "empty_content"},
            }
        result = call_gemini_stream_collect(
            args.model,
            prompt_text,
            retries=max(0, args.request_retries),
            sleep_sec=max(0.0, args.retry_sleep_seconds),
            progress_label="",
            timeout_seconds=args.request_timeout_seconds,
        )
        text = str(result.get("text", "") or result.get("content", "") or "")
        return {
            "status": result.get("status", "success" if text.strip() else "empty_content"),
            "text": text,
            "usage": result.get("usage", {}),
            "request_transport": "stream_collect",
            "call_result": result,
        }
    if args.llm_backend == "ollama":
        started = time.time()
        request_payload = {
            "model": args.model,
            "prompt": prompt_text,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": args.max_output_tokens,
                "num_ctx": args.ollama_num_ctx,
            },
        }
        if args.ollama_format_json:
            request_payload["format"] = "json"
        response = requests.post(
            f"{args.ollama_base_url.rstrip('/')}/api/generate",
            json=request_payload,
            timeout=args.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("response", "") or "")
        return {
            "status": "success" if text.strip() else "empty_content",
            "text": text,
            "usage": {
                "prompt_eval_count": payload.get("prompt_eval_count", ""),
                "eval_count": payload.get("eval_count", ""),
            },
            "request_transport": "ollama_generate",
            "call_result": {
                "status": "success" if text.strip() else "empty_content",
                "elapsed_seconds": round(time.time() - started, 3),
                "done_reason": payload.get("done_reason", ""),
                "total_duration": payload.get("total_duration", ""),
                "load_duration": payload.get("load_duration", ""),
                "prompt_eval_count": payload.get("prompt_eval_count", ""),
                "eval_count": payload.get("eval_count", ""),
            },
        }
    raise ValueError(f"Unsupported backend: {args.llm_backend}")


def load_gt_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    for row in load_tsv(path):
        key = key_of(row)
        if not key:
            continue
        try:
            counts[key] = int(str(row.get("gt_count", "")).strip())
        except ValueError:
            continue
    return counts


def write_run_context(out_dir: Path, args: argparse.Namespace, summary_rows: list[dict[str, Any]]) -> None:
    context = out_dir / "RUN_CONTEXT.md"
    total_included = sum(int(row.get("included_count", 0) or 0) for row in summary_rows)
    total_gt = sum(int(row.get("gt_count", 0) or 0) for row in summary_rows if str(row.get("gt_count", "")))
    count_matches = sum(1 for row in summary_rows if row.get("count_status") == "match")
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        f"generated_at: {now_iso()}",
        f"generated_by: `src/stage2_sampling_labels/build_formulation_universe_discovery_v1.py`",
        f"artifact_version: `{ARTIFACT_VERSION}`",
        f"llm_backend: `{args.llm_backend}`",
        f"model: `{args.model if args.llm_backend != 'none' else ''}`",
        "benchmark_valid: `no`",
        "compare_mode: `diagnostic-only, not benchmark-valid final output`",
        "",
        "Boundary:",
        "- This is a Stage2 formulation-universe discovery diagnostic surface.",
        "- It may propose a frozen formulation row universe for review.",
        "- It is not a completed Stage2 artifact, not a Stage3 input, and not a Stage5 final table.",
        "- Later value-binding stages must not create rows from value evidence; suspected missing rows belong in the review queue.",
        "",
        "Outputs:",
        "- `formulation_universe_frozen_v1.tsv`",
        "- `excluded_candidate_ledger_v1.tsv`",
        "- `unresolved_candidate_review_v1.tsv`",
        "- `analysis/formulation_universe_vs_layer1_gt_counts.tsv`",
        "",
        f"paper_count: `{len(summary_rows)}`",
        f"included_formulation_count: `{total_included}`",
        f"gt_count_sum_when_available: `{total_gt}`",
        f"paper_count_matches_when_gt_available: `{count_matches}`",
    ]
    context.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.manifest_tsv:
        raise SystemExit("--manifest-tsv is required at least once")
    out_dir = repo_path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory exists and is not empty: {out_dir}. Use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["source_packages", "prompts", "raw_responses", "parsed", "analysis"]:
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    manifest_paths = [repo_path(path) for path in args.manifest_tsv]
    current_rows = load_current_rows(repo_path(args.manifest_current_tsv))
    scope_rows = merge_scope_rows(manifest_paths, current_rows)
    if args.paper_key:
        wanted = set(args.paper_key)
        scope_rows = [row for row in scope_rows if key_of(row) in wanted]
    if not scope_rows:
        raise SystemExit("No scope rows selected.")

    gt_counts = load_gt_counts(repo_path(args.gt_counts_tsv)) if args.gt_counts_tsv else {}
    included_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for index, row in enumerate(scope_rows, start=1):
        paper_key = key_of(row)
        package = build_source_package(row, args)
        package_path = out_dir / "source_packages" / f"{paper_key}__source_package.json"
        package_for_disk = {k: v for k, v in package.items() if k not in {"source_text", "table_snippets"}}
        package_for_disk["source_text_char_count"] = len(package["source_text"])
        package_for_disk["table_snippets_char_count"] = len(package["table_snippets"])
        package_path.write_text(json.dumps(package_for_disk, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        prompt_text = build_prompt(package)
        prompt_path = out_dir / "prompts" / f"{paper_key}__formulation_universe_prompt.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")

        raw_path = out_dir / "raw_responses" / f"{paper_key}__formulation_universe_raw_response.json"
        parsed_path = out_dir / "parsed" / f"{paper_key}__formulation_universe.json"
        status = "not_called"
        error = ""
        content = ""
        call_result: dict[str, Any] = {}
        payload: dict[str, Any] = {
            "paper_key": paper_key,
            "included_formulations": [],
            "excluded_candidates": [],
            "unresolved_candidate_review": [],
            "audit_notes": ["No live LLM call was requested."],
        }
        try:
            call_result = call_llm(args, prompt_text)
            content = str(call_result.get("text", call_result.get("content", "")) or "")
            raw_path.write_text(
                json.dumps(
                    {
                        "paper_key": paper_key,
                        "generated_at": now_iso(),
                        "llm_backend": args.llm_backend,
                        "model": args.model if args.llm_backend != "none" else "",
                        "prompt_sha256": stable_hash(prompt_text),
                        "content": content,
                        "call_result": call_result,
                        "usage": call_result.get("usage", {}),
                        "request_transport": call_result.get("request_transport", ""),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if args.llm_backend != "none":
                if not content.strip():
                    status = str(call_result.get("status") or "empty_content")
                    raise ValueError(f"LLM returned empty content; status={status}")
                payload = extract_json_object(content)
                status = "parsed"
            else:
                status = "prompt_only"
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error = str(exc)
            payload["audit_notes"] = [f"LLM call or JSON parsing failed: {error}"]
            if not raw_path.exists():
                raw_path.write_text(
                    json.dumps(
                        {
                            "paper_key": paper_key,
                            "generated_at": now_iso(),
                            "llm_backend": args.llm_backend,
                            "model": args.model if args.llm_backend != "none" else "",
                            "prompt_sha256": stable_hash(prompt_text),
                            "content": content,
                            "call_result": call_result,
                            "error": error,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

        parsed_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paper_included = normalize_included(paper_key, payload)
        paper_excluded = normalize_excluded(paper_key, payload)
        paper_review = normalize_review(paper_key, payload)
        included_rows.extend(paper_included)
        excluded_rows.extend(paper_excluded)
        review_rows.extend(paper_review)
        gt_count = gt_counts.get(paper_key)
        included_count = len(paper_included)
        count_status = ""
        if gt_count is not None:
            count_status = "match" if included_count == gt_count else "count_delta"
        summary_rows.append(
            {
                "paper_key": paper_key,
                "title": package.get("title", ""),
                "status": status,
                "included_count": included_count,
                "excluded_count": len(paper_excluded),
                "review_count": len(paper_review),
                "gt_count": "" if gt_count is None else gt_count,
                "count_delta": "" if gt_count is None else included_count - gt_count,
                "count_status": count_status,
                "source_text_truncated": package.get("source_text_truncated", ""),
                "tables_truncated": package.get("tables_truncated", ""),
                "error": error,
            }
        )
        print(
            f"[{index}/{len(scope_rows)}] {paper_key}: status={status} included={included_count} "
            f"excluded={len(paper_excluded)} review={len(paper_review)}"
        )
        if args.sleep_seconds and index < len(scope_rows):
            time.sleep(args.sleep_seconds)

    write_tsv(out_dir / "formulation_universe_frozen_v1.tsv", included_rows, INCLUDED_FIELDS)
    write_tsv(out_dir / "excluded_candidate_ledger_v1.tsv", excluded_rows, EXCLUDED_FIELDS)
    write_tsv(out_dir / "unresolved_candidate_review_v1.tsv", review_rows, REVIEW_FIELDS)
    write_tsv(
        out_dir / "analysis" / "formulation_universe_vs_layer1_gt_counts.tsv",
        summary_rows,
        [
            "paper_key",
            "title",
            "status",
            "included_count",
            "excluded_count",
            "review_count",
            "gt_count",
            "count_delta",
            "count_status",
            "source_text_truncated",
            "tables_truncated",
            "error",
        ],
    )
    write_run_context(out_dir, args, summary_rows)
    print(f"wrote {to_repo_rel(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
