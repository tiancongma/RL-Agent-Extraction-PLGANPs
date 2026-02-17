#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from tqdm import tqdm

HAS_GENAI = False
try:
    import google.generativeai as genai  # type: ignore
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False


PROMPT_VERSION = "verifier_v1_evidence_only"
PROMPT_TEMPLATE = (
    "You are a verifier. Use ONLY the evidence text provided.\n"
    "Determine whether extracted_value for field_name is supported by evidence_span_text.\n"
    "Output STRICT JSON only with exactly these keys:\n"
    '{"verdict":"supported|insufficient|contradicted","rationale":"<=20 words"}\n'
    "No extra keys. No markdown. No explanations outside JSON."
)
VALID_VERDICTS = {"supported", "insufficient", "contradicted"}


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _prompt_sha256() -> str:
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def ensure_genai(model: str) -> None:
    if not HAS_GENAI:
        raise RuntimeError("google-generativeai is not installed in this environment.")
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY is missing in environment.")
    genai.configure(api_key=key)
    _ = genai.GenerativeModel(model)


def call_gemini(model: str, prompt: str) -> str:
    mdl = genai.GenerativeModel(model)
    resp = mdl.generate_content(prompt)
    if hasattr(resp, "text") and resp.text:
        return str(resp.text)
    try:
        cand = resp.candidates[0].content.parts[0].text
        if cand:
            return str(cand)
    except Exception:
        pass
    return ""


def _strip_code_fences(s: str) -> str:
    t = s.strip()
    t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _parse_verifier_json(raw_text: str) -> Dict[str, Any]:
    txt = _strip_code_fences(raw_text or "")
    try:
        obj = json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            raise ValueError("invalid_json")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("json_not_object")
    verdict = str(obj.get("verdict", "")).strip().lower()
    rationale = str(obj.get("rationale", "")).strip()
    if verdict not in VALID_VERDICTS:
        raise ValueError("missing_or_invalid_verdict")
    if rationale == "":
        raise ValueError("missing_rationale")
    return {"verdict": verdict, "rationale": rationale}


def _build_prompt(field_name: str, extracted_value: str, evidence_span_text: str) -> str:
    return (
        f"{PROMPT_TEMPLATE}\n\n"
        f"field_name: {field_name}\n"
        f"extracted_value: {extracted_value}\n"
        f"evidence_span_text:\n{evidence_span_text}"
    )


def _load_existing_ids(path: Path) -> Set[str]:
    done: Set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rid = str(obj.get("request_id", "")).strip()
            if rid:
                done.add(rid)
    return done


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run verifier requests (JSONL) via Gemini and write responses JSONL.")
    p.add_argument("--in-requests", required=True)
    p.add_argument("--out-responses", required=True)
    p.add_argument("--model", default="gemma-3-12b-it")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sleep", type=float, default=6.0)
    p.add_argument("--limit-requests", type=int, default=0)
    p.add_argument("--resume", dest="resume", action="store_true")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    in_requests = Path(args.in_requests)
    out_responses = Path(args.out_responses)

    if not in_requests.exists():
        raise FileNotFoundError(f"Input requests JSONL not found: {in_requests}")

    ensure_genai(args.model)
    out_responses.parent.mkdir(parents=True, exist_ok=True)

    existing_ids: Set[str] = set()
    if args.resume:
        existing_ids = _load_existing_ids(out_responses)

    mode = "a" if args.resume and out_responses.exists() else "w"
    prompt_sha = _prompt_sha256()

    n_seen = 0
    n_skipped = 0
    n_written = 0

    requests_to_run: List[Dict[str, Any]] = []
    with in_requests.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_seen += 1
            if args.limit_requests and args.limit_requests > 0 and n_seen > args.limit_requests:
                break

            req = json.loads(line)
            req_id = str(req.get("request_id", "")).strip()
            if req_id == "":
                continue
            if args.resume and req_id in existing_ids:
                n_skipped += 1
                continue
            requests_to_run.append(req)

    with out_responses.open(mode, encoding="utf-8") as fout:
        for req in tqdm(requests_to_run, total=len(requests_to_run), desc="Verifier", ncols=80):
            req_id = str(req.get("request_id", "")).strip()

            field_name = str(req.get("field_name", ""))
            extracted_value = str(req.get("extracted_value", ""))
            evidence_span_text = str(req.get("evidence_span_text", ""))
            request_prompt_sha = str(req.get("prompt_sha256", "")).strip() or prompt_sha
            run_id = str(req.get("run_id", "")).strip()

            raw_text = ""
            verdict = None
            rationale = None
            error_message = None
            status = "ok"

            try:
                prompt = _build_prompt(field_name, extracted_value, evidence_span_text)
                raw_text = call_gemini(args.model, prompt)
                parsed = _parse_verifier_json(raw_text)
                verdict = parsed["verdict"]
                rationale = parsed["rationale"]
            except Exception as e:
                status = "error"
                error_message = f"{type(e).__name__}: {e}"

            out = {
                "request_id": req_id,
                "run_id": run_id,
                "status": status,
                "verdict": verdict,
                "rationale": rationale,
                "verifier_model": args.model,
                "temperature": float(args.temperature),
                "prompt_sha256": request_prompt_sha,
                "raw_text": raw_text,
                "error_message": error_message,
            }
            fout.write(_json_dumps(out) + "\n")
            fout.flush()
            n_written += 1
            time.sleep(float(args.sleep))

    print(f"[OK] seen={n_seen}")
    print(f"[OK] skipped_existing={n_skipped}")
    print(f"[OK] written={n_written} -> {out_responses}")


if __name__ == "__main__":
    main()
