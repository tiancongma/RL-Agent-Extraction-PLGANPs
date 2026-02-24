#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regression check: explicit formulation IDs should prevent schema_v2 core collapse."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--doc-key", default="UFXX9WXE")
    parser.add_argument("--before-core-count", type=int, default=-1)
    parser.add_argument(
        "--out-tsv",
        default="",
        help="Default: data/results/<run_id>/debug/schema_v2_explicit_id_regression__<doc_key>.tsv",
    )
    return parser.parse_args()


def normalize_formulation_id_token(value: str) -> str:
    s = str(value).strip().upper()
    if not s:
        return ""
    s = re.sub(r"[^A-Z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("_-")


def has_explicit_formulation_id(value: str) -> bool:
    tok = normalize_formulation_id_token(value)
    return bool(tok and re.search(r"[A-Z]", tok) and re.search(r"\d", tok))


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    doc_key = args.doc_key
    out_tsv = (
        Path(args.out_tsv)
        if args.out_tsv
        else Path(f"data/results/{run_id}/debug/schema_v2_explicit_id_regression__{doc_key}.tsv")
    )
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    base = Path(f"data/results/{run_id}/benchmark_goren_2025/schema_v2")
    trace = pd.read_csv(base / "core_assignment_trace.tsv", sep="\t", dtype=str).fillna("")
    trace["key"] = trace["group_key"].astype(str).str.split("::").str[0]
    trace_doc = trace[trace["key"] == doc_key].copy()
    trace_doc["formulation_id_raw"] = trace_doc["group_key"].astype(str).str.split("::", n=1).str[1].fillna("")
    trace_doc["formulation_id_token"] = trace_doc["formulation_id_raw"].map(normalize_formulation_id_token)
    trace_doc["is_explicit_formulation_id"] = trace_doc["formulation_id_raw"].map(has_explicit_formulation_id)

    after_core_count = int(trace_doc["formulation_core_id"].nunique())
    explicit = trace_doc[trace_doc["is_explicit_formulation_id"]].copy()
    distinct_explicit_ids = int(explicit["formulation_id_token"].nunique())
    acceptance_pass = after_core_count > 1

    detail = (
        explicit[
            [
                "group_key",
                "formulation_id_raw",
                "formulation_id_token",
                "formulation_core_id",
                "core_signature",
            ]
        ]
        .drop_duplicates()
        .sort_values(["formulation_id_token", "formulation_core_id"])
        .reset_index(drop=True)
    )
    if detail.empty:
        detail = pd.DataFrame(
            columns=[
                "group_key",
                "formulation_id_raw",
                "formulation_id_token",
                "formulation_core_id",
                "core_signature",
            ]
        )

    detail.insert(0, "run_id", run_id)
    detail.insert(1, "doc_key", doc_key)
    detail.insert(2, "before_core_count", int(args.before_core_count))
    detail.insert(3, "after_core_count", after_core_count)
    detail.insert(4, "distinct_explicit_formulation_ids", distinct_explicit_ids)
    detail.insert(5, "acceptance_pass", int(acceptance_pass))
    detail.to_csv(out_tsv, sep="\t", index=False)

    print(f"doc_key={doc_key}")
    print(f"before_core_count={int(args.before_core_count)}")
    print(f"after_core_count={after_core_count}")
    print(f"distinct_explicit_formulation_ids={distinct_explicit_ids}")
    print(f"acceptance_pass={int(acceptance_pass)}")
    print(f"output={out_tsv}")


if __name__ == "__main__":
    main()

