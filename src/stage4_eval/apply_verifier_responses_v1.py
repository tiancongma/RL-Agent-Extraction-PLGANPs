#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


VERDICT_VALUES = {"supported", "insufficient", "contradicted", "error", "missing"}


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


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


def _iter_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply verifier responses + rule gates back to extractor TSV.")
    p.add_argument("--in-tsv", required=True)
    p.add_argument("--in-responses", required=True)
    p.add_argument("--in-gates", required=True)
    p.add_argument("--out-tsv", required=True)
    p.add_argument("--out-conflicts", required=True)
    p.add_argument("--out-summary", required=True)
    p.add_argument("--policy", default="hybrid_v1")
    p.add_argument("--verifier-model", default="gemma-3-12b-it")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    in_tsv = Path(args.in_tsv)
    in_responses = Path(args.in_responses)
    in_gates = Path(args.in_gates)
    out_tsv = Path(args.out_tsv)
    out_conflicts = Path(args.out_conflicts)
    out_summary = Path(args.out_summary)

    if not in_tsv.exists():
        raise FileNotFoundError(f"Input TSV not found: {in_tsv}")

    responses = _read_jsonl(in_responses)
    gates = _read_jsonl(in_gates)

    run_ids: Set[str] = set()
    for obj in responses + gates:
        rid = _clean_str(obj.get("run_id"))
        if rid:
            run_ids.add(rid)
    if len(run_ids) == 0:
        raise RuntimeError("Unable to infer run_id from responses/gates JSONL.")
    run_id = sorted(run_ids)[0]

    gate_by_req: Dict[str, Dict[str, Any]] = {}
    for g in gates:
        req_id = _clean_str(g.get("request_id"))
        if req_id:
            gate_by_req[req_id] = g

    resp_by_req: Dict[str, Dict[str, Any]] = {}
    for r in responses:
        req_id = _clean_str(r.get("request_id"))
        if req_id:
            resp_by_req[req_id] = r

    rows = list(_iter_rows(in_tsv))
    if len(rows) == 0:
        raise RuntimeError("Input TSV has no rows.")

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_conflicts.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    base_cols = list(rows[0].keys())
    meta_cols = ["verifier_policy", "verifier_model", "verdict_json", "verdict_source_json"]
    out_cols = base_cols + [c for c in meta_cols if c not in base_cols]

    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "n_total": 0,
        "n_rule": 0,
        "n_llm": 0,
        "n_supported": 0,
        "n_insufficient": 0,
        "n_contradicted": 0,
        "n_error": 0,
        "n_missing": 0,
    })

    verified_rows: List[Dict[str, str]] = []
    conflict_rows: List[Dict[str, str]] = []

    for row in rows:
        fe = _parse_field_evidence_json(_clean_str(row.get("field_evidence_json")))
        key = _clean_str(row.get("key"))
        formulation_id = _clean_str(row.get("formulation_id"))
        text_sha256 = _clean_str(row.get("text_sha256"))

        verdict_map: Dict[str, str] = {}
        source_map: Dict[str, str] = {}

        for field_name in fe.keys():
            _, span_start, span_end = _first_span(fe.get(field_name))
            extracted_value = _clean_str(row.get(field_name))
            req_id = _request_id(
                run_id=run_id,
                key=key,
                formulation_id=formulation_id,
                field_name=field_name,
                extracted_value=extracted_value,
                text_sha256=text_sha256,
                span_start=span_start,
                span_end=span_end,
            )

            verdict = "missing"
            source = "missing"

            if req_id in gate_by_req:
                verdict = "insufficient"
                source = "rule"
            else:
                r = resp_by_req.get(req_id)
                if r is not None:
                    status = _clean_str(r.get("status")).lower()
                    if status == "ok":
                        v = _clean_str(r.get("verdict")).lower()
                        verdict = v if v in VERDICT_VALUES else "error"
                        source = "llm" if verdict != "error" else "error"
                    elif status == "error":
                        verdict = "error"
                        source = "error"

            verdict_map[field_name] = verdict
            source_map[field_name] = source

            s = summary[field_name]
            s["n_total"] += 1
            if source == "rule":
                s["n_rule"] += 1
            if source == "llm":
                s["n_llm"] += 1
            if verdict == "supported":
                s["n_supported"] += 1
            elif verdict == "insufficient":
                s["n_insufficient"] += 1
            elif verdict == "contradicted":
                s["n_contradicted"] += 1
            elif verdict == "error":
                s["n_error"] += 1
            elif verdict == "missing":
                s["n_missing"] += 1

        out_row = dict(row)
        out_row["verifier_policy"] = args.policy
        out_row["verifier_model"] = args.verifier_model
        out_row["verdict_json"] = _json_dumps(verdict_map)
        out_row["verdict_source_json"] = _json_dumps(source_map)
        verified_rows.append(out_row)

        if any(v == "contradicted" for v in verdict_map.values()):
            conflict_rows.append(out_row)

    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in verified_rows:
            w.writerow(r)

    with out_conflicts.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in conflict_rows:
            w.writerow(r)

    summary_cols = [
        "field_name",
        "n_total",
        "n_rule",
        "n_llm",
        "n_supported",
        "n_insufficient",
        "n_contradicted",
        "n_error",
        "n_missing",
        "pct_supported",
        "pct_insufficient",
        "pct_contradicted",
        "pct_error",
        "pct_missing",
    ]
    with out_summary.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols, delimiter="\t")
        w.writeheader()
        for field_name in sorted(summary.keys()):
            s = summary[field_name]
            n_total = max(1, int(s["n_total"]))
            row = {
                "field_name": field_name,
                **s,
                "pct_supported": round(100.0 * s["n_supported"] / n_total, 2),
                "pct_insufficient": round(100.0 * s["n_insufficient"] / n_total, 2),
                "pct_contradicted": round(100.0 * s["n_contradicted"] / n_total, 2),
                "pct_error": round(100.0 * s["n_error"] / n_total, 2),
                "pct_missing": round(100.0 * s["n_missing"] / n_total, 2),
            }
            w.writerow(row)

    print(f"[OK] verified_rows={len(verified_rows)} -> {out_tsv}")
    print(f"[OK] conflicts_rows={len(conflict_rows)} -> {out_conflicts}")
    print(f"[OK] summary_fields={len(summary)} -> {out_summary}")


if __name__ == "__main__":
    main()

