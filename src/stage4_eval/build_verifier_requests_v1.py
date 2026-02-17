#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


NUMERIC_LIKE_FIELDS = {
    "size_nm",
    "pdi",
    "zeta_mV",
    "plga_mass_mg",
    "plga_mw_kDa",
    "pva_conc_percent",
    "encapsulation_efficiency_percent",
    "loading_content_percent",
}

PROMPT_VERSION = "verifier_v1_evidence_only"
PROMPT_TEMPLATE = (
    "You are a verifier. Use ONLY the evidence text provided.\n"
    "Decide whether extracted_value for field_name is supported by evidence_span_text.\n"
    "Return STRICT JSON only with exactly these keys:\n"
    '{"verdict":"supported|insufficient|contradicted","rationale":"<=20 words"}\n'
    "Do not output markdown, code fences, or extra keys."
)
EXPECTED_SCHEMA = {
    "type": "object",
    "required": ["verdict", "rationale"],
    "properties": {
        "verdict": {"type": "string", "enum": ["supported", "insufficient", "contradicted"]},
        "rationale": {"type": "string"},
    },
}


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _prompt_sha256() -> str:
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def _clean_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_int(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s == "":
        return ""
    return s


def _parse_field_evidence_json(s: str) -> Dict[str, Any]:
    raw = _clean_str(s)
    if raw == "":
        return {}
    try:
        obj = json.loads(raw)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _first_span(field_obj: Any) -> Tuple[str, str, str]:
    if not isinstance(field_obj, dict):
        return "", "", ""
    spans = field_obj.get("spans")
    if not isinstance(spans, list) or len(spans) == 0 or not isinstance(spans[0], dict):
        return "", "", ""
    s0 = spans[0]
    return _clean_str(s0.get("text")), _safe_int(s0.get("start")), _safe_int(s0.get("end"))


def _request_id(
    run_id: str,
    key: str,
    formulation_id: str,
    field_name: str,
    extracted_value: str,
    text_sha256: str,
    span_start: str,
    span_end: str,
) -> str:
    payload = "|".join([
        _clean_str(run_id),
        _clean_str(key),
        _clean_str(formulation_id),
        _clean_str(field_name),
        _clean_str(extracted_value),
        _clean_str(text_sha256),
        _clean_str(span_start),
        _clean_str(span_end),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_for_contains(s: str) -> str:
    out = _clean_str(s).lower()
    out = re.sub(r"[\s,\t\r\n:;()\[\]{}\"'`]+", "", out)
    return out


def _value_in_evidence(extracted_value: str, evidence_span_text: str) -> bool:
    v = _normalize_for_contains(extracted_value)
    e = _normalize_for_contains(evidence_span_text)
    if v == "":
        return False
    return v in e


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build verifier request queue JSONL and rule-gate JSONL from extractor TSV.")
    p.add_argument("--in-tsv", required=True)
    p.add_argument("--out-requests", required=True)
    p.add_argument("--out-gates", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--policy", default="hybrid_v1")
    p.add_argument("--verifier-model", default="gemma-3-12b-it")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--fields", default=None, help="Comma-separated subset of fields to verify.")
    p.add_argument("--limit-rows", type=int, default=0)
    p.add_argument("--limit-requests", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _iter_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    in_tsv = Path(args.in_tsv)
    out_requests = Path(args.out_requests)
    out_gates = Path(args.out_gates)

    if not in_tsv.exists():
        raise FileNotFoundError(f"Input TSV not found: {in_tsv}")

    out_requests.parent.mkdir(parents=True, exist_ok=True)
    out_gates.parent.mkdir(parents=True, exist_ok=True)

    selected_fields: Optional[List[str]] = None
    if args.fields and str(args.fields).strip():
        selected_fields = [x.strip() for x in str(args.fields).split(",") if x.strip()]

    random.seed(int(args.seed))
    prompt_sha = _prompt_sha256()

    n_rows = 0
    n_requests = 0
    n_gates = 0

    with out_requests.open("w", encoding="utf-8") as fq, out_gates.open("w", encoding="utf-8") as fg:
        for row in _iter_rows(in_tsv):
            if args.limit_rows and args.limit_rows > 0 and n_rows >= args.limit_rows:
                break
            n_rows += 1

            key = _clean_str(row.get("key"))
            formulation_id = _clean_str(row.get("formulation_id"))
            text_sha256 = _clean_str(row.get("text_sha256"))
            fe = _parse_field_evidence_json(_clean_str(row.get("field_evidence_json")))

            fields_for_row = list(fe.keys())
            if selected_fields is not None:
                fields_for_row = [f for f in selected_fields if f in fe]

            for field_name in fields_for_row:
                if args.limit_requests and args.limit_requests > 0 and n_requests >= args.limit_requests:
                    break

                field_obj = fe.get(field_name)
                evidence_span_text, span_start, span_end = _first_span(field_obj)
                extracted_value = _clean_str(row.get(field_name))
                v_norm = extracted_value.strip().lower()
                if v_norm == "" or v_norm == "nan":
                    # Empty extraction should remain missing in apply step.
                    continue

                req_id = _request_id(
                    run_id=args.run_id,
                    key=key,
                    formulation_id=formulation_id,
                    field_name=field_name,
                    extracted_value=extracted_value,
                    text_sha256=text_sha256,
                    span_start=span_start,
                    span_end=span_end,
                )

                if field_name in NUMERIC_LIKE_FIELDS and not _value_in_evidence(extracted_value, evidence_span_text):
                    gate = {
                        "request_id": req_id,
                        "run_id": args.run_id,
                        "key": key,
                        "formulation_id": formulation_id,
                        "field_name": field_name,
                        "extracted_value": extracted_value,
                        "verdict": "insufficient",
                        "source": "rule",
                        "reason": "value_not_in_evidence",
                    }
                    fg.write(_json_dumps(gate) + "\n")
                    n_gates += 1
                    continue

                req = {
                    "request_id": req_id,
                    "run_id": args.run_id,
                    "key": key,
                    "formulation_id": formulation_id,
                    "field_name": field_name,
                    "extracted_value": extracted_value,
                    "evidence_span_text": evidence_span_text,
                    "text_sha256": text_sha256,
                    "verifier_model": args.verifier_model,
                    "temperature": float(args.temperature),
                    "policy": args.policy,
                    "prompt_version": PROMPT_VERSION,
                    "prompt_sha256": prompt_sha,
                    "expected_schema": EXPECTED_SCHEMA,
                    "span_start": span_start,
                    "span_end": span_end,
                }
                fq.write(_json_dumps(req) + "\n")
                n_requests += 1

            if args.limit_requests and args.limit_requests > 0 and n_requests >= args.limit_requests:
                break

    print(f"[OK] rows_processed={n_rows}")
    print(f"[OK] requests_written={n_requests} -> {out_requests}")
    print(f"[OK] gates_written={n_gates} -> {out_gates}")


if __name__ == "__main__":
    main()
