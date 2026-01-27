#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
sample10_from_zotero_manifest.py

Usage (PowerShell, from project root):

    .\.venv\Scripts\python.exe .\scripts\sample10_from_zotero_manifest.py ^
        --manifest .\data\cleaned\manifests\zotero_llm_relevant.jsonl ^
        --out-manifest .\data\cleaned\samples\manifest_html10.tsv ^
        --out-dir .\data\cleaned\samples\text_html10 ^
        --n 10 ^
        --seed 123

功能概述：
- 从 JSONL manifest (zotero_llm_relevant.jsonl) 中筛选 HTML + OK 的条目；
- 随机抽样，做“小批量体检”：
  - 检查 cleaned text 文件是否存在；
  - 文件大小是否 ≥ --min-bytes（默认 1500 bytes）；
  - 文本长度是否 ≥ --min-chars（默认 200 字符）；
- 通过检查的样本：
  - 复制 cleaned text 到 out-dir；
  - 写入 GOOD manifest (out-manifest)；
- 未通过检查的样本：
  - 不复制，不进入 GOOD manifest；
  - 写入一个单独的 BAD manifest: <out-manifest stem>_bad.tsv；
- 如果候选用完还不足 N 篇 GOOD，会在终端给出 WARN。
"""

import argparse
import json
import pathlib
import random
import shutil
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

# Project root = repo root (…/RL-Agent-Extraction-PLGANPs)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Sample N GOOD HTML entries from zotero_llm_relevant.jsonl "
            "(checking cleaned text size/length) and copy their cleaned text files."
        )
    )
    p.add_argument(
        "--manifest",
        type=pathlib.Path,
        required=True,
        help=(
            "Path to zotero_llm_relevant.jsonl "
            "(e.g. data/cleaned/manifests/zotero_llm_relevant.jsonl)"
        ),
    )
    p.add_argument(
        "--out-manifest",
        type=pathlib.Path,
        required=True,
        help=(
            "Path to write the SMALL sampled TSV manifest, "
            "e.g. data/cleaned/samples/manifest_html10.tsv"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        required=True,
        help=(
            "Directory to copy cleaned text files into, "
            "e.g. data/cleaned/samples/text_html10"
        ),
    )
    p.add_argument(
        "--n",
        type=int,
        default=10,
        help="Target number of GOOD rows to sample (default: 10).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible sampling (default: 123).",
    )
    p.add_argument(
        "--min-bytes",
        type=int,
        default=1500,
        help="Minimum file size (bytes) for cleaned text to be considered valid.",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Minimum non-whitespace character length for cleaned text.",
    )
    return p


def load_manifest_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load JSONL manifest into a list of dictionaries."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Failed to parse line as JSON: {e}")
    return rows


def resolve_text_path(text_path_str: str) -> pathlib.Path:
    """
    Interpret paths.text either as absolute or relative to PROJECT_ROOT.
    """
    p = pathlib.Path(text_path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def check_text_file(
    text_path: pathlib.Path,
    min_bytes: int,
    min_chars: int,
) -> Tuple[bool, str, Optional[int], Optional[int]]:
    """
    Check a cleaned text file for basic quality.

    Returns:
        (ok, reason, size_bytes, text_len)
        - ok: True if passes all checks
        - reason: explanation if not ok ("missing", "too_small_bytes", "too_short_text", ...)
        - size_bytes: file size in bytes (if known)
        - text_len: text length after strip (if known)
    """
    if not text_path.exists():
        return False, "missing_text_file", None, None

    size_bytes = text_path.stat().st_size
    if size_bytes == 0:
        return False, "size_0", size_bytes, None
    if size_bytes < min_bytes:
        return False, f"too_small_bytes(<{min_bytes})", size_bytes, None

    try:
        with text_path.open("r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        return False, f"read_error:{e}", size_bytes, None

    text_len = len(text.strip())
    if text_len < min_chars:
        return False, f"too_short_text(<{min_chars})", size_bytes, text_len

    return True, "OK", size_bytes, text_len


def main() -> None:
    args = build_arg_parser().parse_args()

    manifest_path: pathlib.Path = args.manifest
    out_manifest_path: pathlib.Path = args.out_manifest
    out_dir: pathlib.Path = args.out_dir
    bad_manifest_path: pathlib.Path = out_manifest_path.with_name(
        f"{out_manifest_path.stem}_bad.tsv"
    )

    if not manifest_path.exists():
        raise SystemExit(f"[ERROR] Manifest JSONL not found: {manifest_path}")

    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading JSONL manifest from: {manifest_path}")
    records = load_manifest_jsonl(manifest_path)
    if not records:
        raise SystemExit("[ERROR] No records found in JSONL manifest.")

    # Convert to DataFrame for easy filtering.
    df = pd.json_normalize(records)

    # Expected columns:
    #   - source_type (html / pdf / None)
    #   - status (OK_HTML, OK_PDF, SKIP, FAIL, ...)
    #   - paths.text (cleaned text path)
    if "source_type" not in df.columns or "status" not in df.columns:
        raise SystemExit(
            "[ERROR] JSONL manifest is missing 'source_type' or 'status'."
        )

    # Filter: HTML + successfully cleaned (status starts with 'OK_')
    mask_html = df["source_type"].fillna("").str.lower().eq("html")
    mask_ok = df["status"].fillna("").str.startswith("OK_")
    df_filtered = df[mask_html & mask_ok].copy()

    print(f"[INFO] Total rows in manifest: {len(df)}")
    print(f"[INFO] Rows with HTML + OK status: {len(df_filtered)}")

    if df_filtered.empty:
        raise SystemExit("[ERROR] No HTML+OK rows found in manifest.")

    # Shuffle candidates with seed
    random.seed(args.seed)
    candidate_idx: List[int] = list(df_filtered.index)
    random.shuffle(candidate_idx)

    target_n = args.n
    good_rows: List[Dict[str, Any]] = []
    good_copied_paths: List[str] = []
    bad_rows: List[Dict[str, Any]] = []

    print(
        f"[INFO] Sampling up to {target_n} GOOD rows "
        f"(min_bytes={args.min_bytes}, min_chars={args.min_chars}) "
        f"from {len(candidate_idx)} HTML+OK candidates."
    )

    for idx in candidate_idx:
        if len(good_rows) >= target_n:
            break

        row = df_filtered.loc[idx]
        text_path_str = (row.get("paths.text") or "").strip()

        if not text_path_str:
            reason = "missing_paths.text"
            print(f"[WARN] Row {idx}: {reason}")
            bad_rows.append(
                {
                    "zotero_key": row.get("zotero_key", ""),
                    "title": row.get("title", ""),
                    "year": row.get("year", ""),
                    "doi": row.get("doi", ""),
                    "source_type": row.get("source_type", ""),
                    "status": row.get("status", ""),
                    "paths.text": "",
                    "bad_reason": reason,
                    "size_bytes": None,
                    "text_len": None,
                }
            )
            continue

        src = resolve_text_path(text_path_str)
        ok, reason, size_bytes, text_len = check_text_file(
            src, args.min_bytes, args.min_chars
        )

        if not ok:
            print(
                f"[WARN] Row {idx}: text file failed quality check "
                f"({reason}), path={src}, size={size_bytes}, len={text_len}"
            )
            bad_rows.append(
                {
                    "zotero_key": row.get("zotero_key", ""),
                    "title": row.get("title", ""),
                    "year": row.get("year", ""),
                    "doi": row.get("doi", ""),
                    "source_type": row.get("source_type", ""),
                    "status": row.get("status", ""),
                    "paths.text": text_path_str,
                    "bad_reason": reason,
                    "size_bytes": size_bytes,
                    "text_len": text_len,
                }
            )
            continue

        # GOOD → copy
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        print(f"[INFO] [GOOD] Copied: {src} -> {dst}")

        dst_abs = dst.resolve()
        try:
            rel = dst_abs.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel = str(dst_abs)

        good_rows.append(row.to_dict())
        good_copied_paths.append(rel)

    print(
        f"[INFO] Completed scan: GOOD={len(good_rows)}, BAD={len(bad_rows)}, "
        f"target={target_n}"
    )
    if len(good_rows) < target_n:
        print(
            f"[WARN] Only {len(good_rows)} GOOD rows found; "
            f"could not reach target N={target_n}."
        )

    # Build SMALL manifest for GOOD entries only
    if good_rows:
        df_good = pd.DataFrame(good_rows)
        small = pd.DataFrame(
            {
                "zotero_key": df_good.get("zotero_key", ""),
                "title": df_good.get("title", ""),
                "year": df_good.get("year", ""),
                "doi": df_good.get("doi", ""),
                "source_type": df_good.get("source_type", ""),
                "status": df_good.get("status", ""),
                "html_path": df_good.get("paths.html", ""),
                "cleaned_text_sample": good_copied_paths,
            }
        )

        print(f"[INFO] Writing GOOD sampled TSV manifest to: {out_manifest_path}")
        small.to_csv(out_manifest_path, sep="\t", index=False)
    else:
        print("[WARN] No GOOD rows to write to sampled manifest.")

    # Write BAD entries manifest (within sampled/checked pool)
    if bad_rows:
        df_bad = pd.DataFrame(bad_rows)
        print(f"[INFO] Writing BAD entries TSV manifest to: {bad_manifest_path}")
        df_bad.to_csv(bad_manifest_path, sep="\t", index=False)
    else:
        print("[INFO] No BAD entries encountered; no bad-manifest written.")

    print("[DONE] Sampled GOOD HTML entries and copied cleaned text files.")


if __name__ == "__main__":
    main()
