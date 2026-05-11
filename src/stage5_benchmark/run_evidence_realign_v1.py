#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"

NUMERIC_FIELDS = [
    "drug_feed_amount_text",
    "plga_mass_mg",
    "pva_conc_percent",
    "encapsulation_efficiency_percent",
    "loading_content_percent",
    "size_nm",
    "polymer_mw_kDa",
    "la_ga_ratio",
]

FIELD_UNITS = {
    "drug_feed_amount_text": ["mg", "g", "ug", "mcg", "ng", "mg/ml", "g/ml", "%"],
    "plga_mass_mg": ["mg", "g", "ug", "mcg", "ng", "mg/ml", "g/ml"],
    "pva_conc_percent": ["%", "percent", "wt%", "w/v", "%w/v"],
    "encapsulation_efficiency_percent": ["%", "percent"],
    "loading_content_percent": ["%", "percent"],
    "size_nm": ["nm", "nanometer", "nanometers"],
    "polymer_mw_kDa": ["kda", "da", "mw"],
    "la_ga_ratio": [":"],
}

TABLE_MARKER_RE = re.compile(r"^===\s*TABLE\s+(\d+)\s*\(TSV\)\s*===", re.IGNORECASE | re.MULTILINE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deterministically realign evidence spans by matching extracted numeric values in full cleaned text."
    )
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--input-tsv", default="", help="Default: data/results/<run_id>/weak_labels__gemini.tsv")
    p.add_argument("--sample-manifest", default="data/cleaned/samples/sample_goren18.tsv")
    p.add_argument("--key2txt", default="data/cleaned/content_goren_2025/key2txt.tsv")
    p.add_argument("--float-match-rel-tol", type=float, default=0.01)
    p.add_argument("--float-match-abs-tol", type=float, default=1e-6)
    p.add_argument("--out-tsv", default="", help="Default: data/results/<run_id>/evidence_realign_log.tsv")
    return p.parse_args()


def canonicalize_num(num: float) -> str:
    if abs(num) < 1e-12:
        return "0"
    return f"{num:.12f}".rstrip("0").rstrip(".")


def parse_numbers(text: str) -> list[float]:
    vals = []
    for tok in re.findall(r"[+-]?\d+(?:\.\d+)?", str(text)):
        try:
            vals.append(float(tok))
        except ValueError:
            continue
    return vals


def get_main_numeric_token(value_text: str) -> tuple[str, float | None]:
    nums = parse_numbers(value_text)
    if not nums:
        return ("", None)
    return (canonicalize_num(nums[0]), nums[0])


def normalize_text(text: str) -> str:
    return (
        str(text)
        .lower()
        .replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2010", "-")
    )


def discover_source_for_key(key: str, sample_manifest: pd.DataFrame, key2txt: pd.DataFrame) -> Path | None:
    key = str(key).strip()
    sub = sample_manifest[sample_manifest["key"].astype(str).str.strip() == key]
    if not sub.empty and "text_path" in sub.columns:
        raw = str(sub.iloc[0]["text_path"]).strip()
        if raw:
            p = Path(raw.replace("\\", "/"))
            if p.exists():
                return p
    if not key2txt.empty:
        sub2 = key2txt[key2txt["key"].astype(str).str.strip() == key]
        if not sub2.empty and "txt_path" in sub2.columns:
            raw2 = str(sub2.iloc[0]["txt_path"]).strip()
            if raw2:
                p2 = Path("data/cleaned/content_goren_2025") / raw2.replace("\\", "/")
                if p2.exists():
                    return p2
    return None


def extract_table_blocks(text: str) -> list[dict[str, Any]]:
    matches = list(TABLE_MARKER_RE.finditer(text))
    blocks: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block_text = text[start:end]
        blocks.append(
            {
                "table_id": m.group(1),
                "start": start,
                "end": end,
                "text": block_text,
            }
        )
    return blocks


def extract_sentences(text: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for m in re.finditer(r"[^.!?\n]+[.!?]?", text):
        s = m.group(0).strip()
        if not s:
            continue
        spans.append({"start": m.start(), "end": m.end(), "text": s})
    return spans


def contains_unit(norm_text: str, unit_tokens: list[str]) -> bool:
    if not unit_tokens:
        return True
    return any(u in norm_text for u in unit_tokens)


def numeric_match(
    norm_text: str,
    main_token: str,
    main_num: float | None,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    if main_token and main_token in norm_text:
        return True
    if main_num is None:
        return False
    nums = parse_numbers(norm_text)
    for n in nums:
        if abs(n - main_num) <= max(abs_tol, abs(main_num) * rel_tol):
            return True
    return False


def first_row_start_in_block(
    block: dict[str, Any],
    main_token: str,
    main_num: float | None,
    unit_tokens: list[str],
    rel_tol: float,
    abs_tol: float,
) -> int | None:
    lines = block["text"].splitlines(keepends=True)
    offset = 0
    for line in lines:
        norm = normalize_text(line)
        if numeric_match(norm, main_token, main_num, rel_tol, abs_tol) and contains_unit(norm, unit_tokens):
            return int(block["start"] + offset)
        offset += len(line)
    return None


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    input_tsv = Path(args.input_tsv) if args.input_tsv else Path(f"data/results/{run_id}/weak_labels__gemini.tsv")
    out_tsv = Path(args.out_tsv) if args.out_tsv else Path(f"data/results/{run_id}/evidence_realign_log.tsv")
    sample_manifest_path = Path(args.sample_manifest)
    key2txt_path = Path(args.key2txt)

    if not input_tsv.exists():
        raise FileNotFoundError(f"Missing input TSV: {input_tsv}")
    if not sample_manifest_path.exists():
        raise FileNotFoundError(f"Missing sample manifest: {sample_manifest_path}")

    df = pd.read_csv(input_tsv, sep="\t", dtype=str).fillna("")
    sample_manifest = pd.read_csv(sample_manifest_path, sep="\t", dtype=str).fillna("")
    key2txt = pd.read_csv(key2txt_path, sep="\t", dtype=str).fillna("") if key2txt_path.exists() else pd.DataFrame()

    source_cache: dict[str, tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = {}
    out_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        key = str(row.get("key", "")).strip()
        formulation_id = str(row.get("formulation_id", "")).strip()
        old_span_start = str(row.get("evidence_span_start", "")).strip()
        if key not in source_cache:
            path = discover_source_for_key(key, sample_manifest, key2txt)
            if path is None:
                source_cache[key] = ("", [], [])
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
                source_cache[key] = (text, extract_table_blocks(text), extract_sentences(text))

        text, table_blocks, sentences = source_cache[key]
        if not text:
            for field in NUMERIC_FIELDS:
                if field not in df.columns:
                    continue
                value_text = str(row.get(field, "")).strip()
                if not value_text:
                    continue
                out_rows.append(
                    {
                        "key": key,
                        "formulation_id": formulation_id,
                        "field_name": field,
                        "old_span_start": old_span_start,
                        "new_span_start": "",
                        "span_type": "",
                        "realign_success": False,
                    }
                )
            continue

        for field in NUMERIC_FIELDS:
            if field not in df.columns:
                continue
            value_text = str(row.get(field, "")).strip()
            if not value_text:
                continue

            main_token, main_num = get_main_numeric_token(value_text)
            if not main_token and main_num is None:
                continue
            unit_tokens = FIELD_UNITS.get(field, [])
            norm_units = [normalize_text(u) for u in unit_tokens if str(u).strip()]

            new_start: int | None = None
            span_type = ""

            # Prefer table blocks.
            for block in table_blocks:
                norm_block = normalize_text(block["text"])
                if not numeric_match(norm_block, main_token, main_num, args.float_match_rel_tol, args.float_match_abs_tol):
                    continue
                if not contains_unit(norm_block, norm_units):
                    continue
                row_start = first_row_start_in_block(
                    block,
                    main_token,
                    main_num,
                    norm_units,
                    args.float_match_rel_tol,
                    args.float_match_abs_tol,
                )
                if row_start is not None:
                    new_start = row_start
                    span_type = "table_row"
                else:
                    new_start = int(block["start"])
                    span_type = "table_block"
                break

            # Fallback to sentence.
            if new_start is None:
                for sent in sentences:
                    norm_sent = normalize_text(sent["text"])
                    if not numeric_match(norm_sent, main_token, main_num, args.float_match_rel_tol, args.float_match_abs_tol):
                        continue
                    if not contains_unit(norm_sent, norm_units):
                        continue
                    new_start = int(sent["start"])
                    span_type = "sentence"
                    break

            out_rows.append(
                {
                    "key": key,
                    "formulation_id": formulation_id,
                    "field_name": field,
                    "old_span_start": old_span_start,
                    "new_span_start": "" if new_start is None else int(new_start),
                    "span_type": span_type,
                    "realign_success": bool(new_start is not None),
                }
            )

    out_df = pd.DataFrame(
        out_rows,
        columns=[
            "key",
            "formulation_id",
            "field_name",
            "old_span_start",
            "new_span_start",
            "span_type",
            "realign_success",
        ],
    ).sort_values(["key", "formulation_id", "field_name"]).reset_index(drop=True)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_tsv, sep="\t", index=False)

    success = int((out_df["realign_success"] == True).sum()) if not out_df.empty else 0
    print(f"output={out_tsv}")
    print(f"rows={len(out_df)}")
    print(f"realign_success={success}")


if __name__ == "__main__":
    main()
