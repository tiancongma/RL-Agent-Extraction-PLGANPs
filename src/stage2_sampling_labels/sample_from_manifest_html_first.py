#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sample_from_manifest_html_first.py

Purpose
- Create a small, reproducible subset (sample10, sample20, sample30, …) from the authoritative manifest TSV.
- Default sampling is HTML-first random (seeded).

Key contracts
- Default paths resolved via src/utils/paths.py (no hard-coded repo paths).
- Sampling operates on keys from manifest_current.tsv, but only admits keys that have an existing cleaned text
  file according to data/cleaned/index/key2txt.tsv (2-column mapping: key -> repo-relative text path).

Modes
- htmlfirst (default): html-first random sample of size --n (seeded). Uses source_type from strata_tags.tsv
  if available; otherwise falls back to manifest columns.
- stratified20: STRICTLY requires data/cleaned/index/strata_tags.tsv.
  Stratification dimensions:
    particle_scale_tag ∈ {nano, micro}
    emulsion_route_tag ∈ {O/W, W/O/W}
    reporting_style_tag ∈ {table, text}
  Additionally supports a soft preference for choosing HTML when available within a candidate set
  via --html-bias (0-1).

Outputs
- JSONL sample manifest (one record per key) and a companion TSV (unless --no-tsv).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

# ---- bootstrap repo root for "import src.*" when running as a script ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths  # noqa: E402


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _pick_id_column(df: pd.DataFrame) -> str:
    for c in ["zotero_key", "paper_id", "key", "item_key", "id"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _read_key2txt_map(map_path: Path) -> Dict[str, Path]:
    """
    Read 2-column mapping: key <tab> path (repo-relative or absolute).
    """
    m: Dict[str, Path] = {}
    with map_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split("\t")
            if len(parts) < 2:
                continue
            k = parts[0].strip()
            p = parts[1].strip()
            if not k or not p:
                continue
            pp = Path(p)
            if not pp.is_absolute():
                pp = (paths.PROJECT_ROOT / pp).resolve()
            m[k] = pp
    return m


def _write_jsonl(records: Sequence[dict], out_path: Path, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {out_path}")
    _ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_tsv(records: Sequence[dict], out_path: Path, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {out_path}")
    _ensure_parent(out_path)
    pd.DataFrame(list(records)).to_csv(out_path, sep="\t", index=False, encoding="utf-8")


def _load_tags(tags_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tags_path, sep="\t", dtype=str, keep_default_na=False).replace({"": pd.NA})
    if "key" not in df.columns:
        raise ValueError(f"strata_tags.tsv missing required column 'key': {tags_path}")
    # Normalize
    df["key"] = df["key"].astype(str)
    # Ensure source_type standardized
    if "source_type" in df.columns:
        df["source_type"] = df["source_type"].astype(str).str.strip().str.upper()
    return df


def _filter_only_with_text(df: pd.DataFrame, id_col: str, key2txt: Dict[str, Path], verbose: bool) -> pd.DataFrame:
    keep_rows = []
    missing = 0
    nofile = 0
    for _, row in df.iterrows():
        pid = str(row[id_col])
        p = key2txt.get(pid)
        if p is None:
            missing += 1
            continue
        if not p.exists():
            nofile += 1
            continue
        keep_rows.append(row)
    if verbose:
        print(f"[info] Filter by existing text: kept={len(keep_rows)} missing_in_key2txt={missing} text_missing_on_disk={nofile}")
    return pd.DataFrame(keep_rows).reset_index(drop=True)


def _coerce_bool(x) -> Optional[bool]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _safe_int(x) -> Optional[int]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def sample_htmlfirst(
    df: pd.DataFrame,
    id_col: str,
    key2txt: Dict[str, Path],
    seed: int,
    n: int,
    tags_df: Optional[pd.DataFrame],
) -> List[dict]:
    """
    HTML-first sampling:
    - Build HTML and PDF pools using source_type from tags when available, else manifest source_type if present.
    - Shuffle each pool with seed.
    - Take HTML first, then PDF, until reaching n.
    """
    rng = random.Random(seed)

    tag_map = None
    if tags_df is not None:
        tag_map = {str(r["key"]): r for _, r in tags_df.iterrows()}

    def _source_type_for_key(k: str, row: pd.Series) -> str:
        if tag_map is not None and k in tag_map:
            st = str(tag_map[k].get("source_type") or "").strip().lower()
            if st:
                return st
        # fallback to manifest
        for c in ["source_type", "source", "preferred_source", "content_type"]:
            if c in row.index and pd.notna(row[c]):
                s = str(row[c]).strip().lower()
                if s:
                    return s
        return ""

    html_rows: List[pd.Series] = []
    pdf_rows: List[pd.Series] = []

    for _, row in df.iterrows():
        k = str(row[id_col])
        st = _source_type_for_key(k, row)
        if "html" in st:
            html_rows.append(row)
        else:
            pdf_rows.append(row)

    rng.shuffle(html_rows)
    rng.shuffle(pdf_rows)

    picked: List[pd.Series] = []
    need = n

    take_html = min(need, len(html_rows))
    if take_html > 0:
        picked.extend(html_rows[:take_html])
        need -= take_html

    if need > 0 and len(pdf_rows) > 0:
        take_pdf = min(need, len(pdf_rows))
        picked.extend(pdf_rows[:take_pdf])
        need -= take_pdf

    out: List[dict] = []
    for row in picked:
        k = str(row[id_col])
        p = key2txt[k]
        out.append({
            "key": k,
            "paper_id": k,
            "text_path": str(p.resolve()),
            "selection_reason": "random_htmlfirst",
        })
    return out


def sample_stratified20(
    df: pd.DataFrame,
    id_col: str,
    key2txt: Dict[str, Path],
    seed: int,
    tags_df: pd.DataFrame,
    verbose: bool,
    html_bias: float,
) -> List[dict]:
    """
    Stratified sample to 20 using precomputed strata_tags.tsv.

    html_bias:
    - Soft preference for selecting HTML when both HTML and PDF candidates exist within a candidate set.
    - 0.0 means no preference (pure random). 1.0 means always prefer HTML when available.
    """
    rng = random.Random(seed)

    # Clamp html_bias to [0, 1]
    if html_bias < 0.0:
        html_bias = 0.0
    if html_bias > 1.0:
        html_bias = 1.0

    # Merge tags onto manifest by key
    work = df.copy()
    work[id_col] = work[id_col].astype(str)
    tags_df = tags_df.copy()
    tags_df["key"] = tags_df["key"].astype(str)
    work = work.merge(tags_df, how="left", left_on=id_col, right_on="key", suffixes=("", "_tag"))

    # Strict requirement: tags must exist
    required_cols = ["particle_scale_tag", "emulsion_route_tag", "reporting_style_tag", "source_type"]
    for c in required_cols:
        if c not in work.columns:
            raise RuntimeError(f"strata_tags.tsv missing required column: {c}")

    def _is_html_row(r: pd.Series) -> bool:
        st = str(r.get("source_type") or "").strip().lower()
        return "html" in st

    def _choose_id_with_html_bias(cand_df: pd.DataFrame, ids: List[str]) -> str:
        if not ids:
            raise ValueError("ids is empty")
        html_ids: List[str] = []
        for pid in ids:
            rr = cand_df[cand_df[id_col].astype(str) == pid]
            if rr.empty:
                continue
            if _is_html_row(rr.iloc[0]):
                html_ids.append(pid)
        if html_ids and rng.random() < html_bias:
            return rng.choice(html_ids)
        return rng.choice(ids)

    used: set[str] = set()
    out: List[dict] = []

    def _add_row(row: pd.Series, reason: str) -> None:
        k = str(row[id_col])
        if k in used:
            return
        p = key2txt.get(k)
        if p is None or not p.exists():
            return
        used.add(k)

        td = _coerce_bool(row.get("table_detected"))
        tl = _safe_int(row.get("text_length"))
        src = str(row.get("source_type") or "").strip().upper()

        out.append({
            "key": k,
            "paper_id": k,
            "text_path": str(p.resolve()),
            "particle_scale_tag": str(row.get("particle_scale_tag") or "unknown"),
            "emulsion_route_tag": str(row.get("emulsion_route_tag") or "unknown"),
            "reporting_style_tag": str(row.get("reporting_style_tag") or "unknown"),
            "source_type": src,
            "text_length": "" if tl is None else tl,
            "table_detected": "" if td is None else (1 if td else 0),
            "parse_quality": str(row.get("parse_quality") or ""),
            "selection_reason": reason,
        })

    # Core 8 strata (note: this is strict to nano/micro, O/W/W/O/W, table/text)
    core_particles = ["nano", "micro"]
    core_emulsions = ["O/W", "W/O/W"]
    core_styles = ["table", "text"]

    for ptag in core_particles:
        for etag in core_emulsions:
            for stag in core_styles:
                cand = work[
                    (work["particle_scale_tag"] == ptag)
                    & (work["emulsion_route_tag"] == etag)
                    & (work["reporting_style_tag"] == stag)
                ]
                if cand.empty:
                    if verbose:
                        print(f"[stratified20] Missing stratum ({ptag}, {etag}, {stag})")
                    continue
                ids = [str(x) for x in cand[id_col].tolist() if str(x) not in used]
                if not ids:
                    continue
                chosen = _choose_id_with_html_bias(cand, ids)
                row = cand[cand[id_col].astype(str) == chosen].iloc[0]
                _add_row(row, "stratum_core")

    # High/low information outliers (optional, uses text_length from tags if present)
    lens = work["text_length"].dropna().astype(str)
    lens_int = []
    for v in lens.tolist():
        iv = _safe_int(v)
        if iv is not None:
            lens_int.append(iv)

    if len(lens_int) >= 10:
        s = pd.Series(lens_int)
        hi_cut = float(s.quantile(0.90))
        lo_cut = float(s.quantile(0.10))

        def _row_len_ge(x, thr: float) -> bool:
            iv = _safe_int(x)
            return iv is not None and float(iv) >= thr

        def _row_len_le(x, thr: float) -> bool:
            iv = _safe_int(x)
            return iv is not None and float(iv) <= thr

        high = work[work["text_length"].apply(lambda x: _row_len_ge(x, hi_cut))]
        low = work[work["text_length"].apply(lambda x: _row_len_le(x, lo_cut))]

        def _pick_k(df_sub: pd.DataFrame, k: int, reason: str) -> None:
            ids2 = [str(x) for x in df_sub[id_col].tolist() if str(x) not in used]
            if not ids2:
                return
            if len(ids2) <= k:
                picks = ids2
            else:
                picks = []
                pool = ids2[:]
                while pool and len(picks) < k:
                    pid = _choose_id_with_html_bias(df_sub, pool)
                    picks.append(pid)
                    pool = [x for x in pool if x != pid]
            for pid in picks:
                row = df_sub[df_sub[id_col].astype(str) == pid].iloc[0]
                _add_row(row, reason)

        _pick_k(high, 2, "high_information")
        _pick_k(low, 2, "low_information")
    else:
        if verbose:
            print("[stratified20] text_length not available; skipping high/low information supplements.")

    # Random fill to 20 (still uses html bias)
    need = 20 - len(out)
    if need > 0:
        candidates = [str(x) for x in work[id_col].astype(str).tolist() if str(x) not in used]
        if candidates:
            if len(candidates) <= need:
                picks = candidates
            else:
                picks = []
                pool = candidates[:]
                while pool and len(picks) < need:
                    pid = _choose_id_with_html_bias(work, pool)
                    picks.append(pid)
                    pool = [x for x in pool if x != pid]
            for pid in picks:
                row = work[work[id_col].astype(str) == pid].iloc[0]
                _add_row(row, "random_fill")

    return out[:20]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build a reproducible sample JSONL from the authoritative manifest.")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=(paths.DATA_CLEANED_INDEX_DIR / "manifest_current.tsv"),
        help="Path to data/cleaned/index/manifest_current.tsv",
    )
    ap.add_argument(
        "--out-jsonl",
        type=Path,
        required=True,
        help="Output JSONL path (e.g., data/cleaned/samples/sample10_htmlfirst.jsonl)",
    )
    ap.add_argument("--n", type=int, default=10, help="Number of items for htmlfirst (default: 10).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    ap.add_argument("--mode", choices=["htmlfirst", "stratified20"], default="htmlfirst", help="Sampling mode.")
    ap.add_argument(
        "--strata-tags",
        type=Path,
        default=(paths.DATA_CLEANED_INDEX_DIR / "strata_tags.tsv"),
        help="Path to strata tag TSV (default: data/cleaned/index/strata_tags.tsv). Required for stratified20.",
    )
    ap.add_argument(
        "--html-bias",
        type=float,
        default=0.8,
        help="Soft preference for choosing HTML when available in stratified20 (0-1, default: 0.8).",
    )
    ap.add_argument("--no-tsv", action="store_true", help="Do not emit a companion TSV next to JSONL.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    df = pd.read_csv(args.manifest, sep="\t", dtype=str, keep_default_na=False).replace({"": pd.NA})
    id_col = _pick_id_column(df)
    if args.verbose:
        print(f"[info] id_col = {id_col}")

    key2txt_path = paths.DATA_CLEANED_INDEX_DIR / "key2txt.tsv"
    if not key2txt_path.exists():
        raise SystemExit(f"key2txt mapping not found: {key2txt_path}")
    key2txt = _read_key2txt_map(key2txt_path)

    df = _filter_only_with_text(df, id_col=id_col, key2txt=key2txt, verbose=args.verbose)
    if df.empty:
        raise SystemExit("No eligible entries with existing text files found (after key2txt filter).")

    tags_df: Optional[pd.DataFrame] = None
    if args.strata_tags.exists():
        tags_df = _load_tags(args.strata_tags)

    if args.mode == "htmlfirst":
        records = sample_htmlfirst(df, id_col=id_col, key2txt=key2txt, seed=args.seed, n=args.n, tags_df=tags_df)
    else:
        if not args.strata_tags.exists():
            raise SystemExit(
                f"stratified20 requires strata tags, but file not found: {args.strata_tags}. "
                f"Run: python src/stage2_sampling_labels/build_strata_tags_from_text.py --overwrite"
            )
        records = sample_stratified20(
            df,
            id_col=id_col,
            key2txt=key2txt,
            seed=args.seed,
            tags_df=tags_df,
            verbose=args.verbose,
            html_bias=args.html_bias,
        )
        if args.verbose and args.n != 20:
            print(f"[info] stratified20 ignores --n={args.n} and targets 20.")

    if args.verbose:
        h = sum(1 for r in records if str(r.get("source_type", "")).strip().upper() == "HTML")
        p = sum(1 for r in records if str(r.get("source_type", "")).strip().upper() == "PDF")
        # Fallback: if source_type missing, count as pdf
        if (h + p) < len(records):
            p += (len(records) - (h + p))
        print(f"[info] Sample size={len(records)} (html={h}, pdf={p})")

    _write_jsonl(records, args.out_jsonl, overwrite=args.overwrite)
    if not args.no_tsv:
        _write_tsv(records, args.out_jsonl.with_suffix(".tsv"), overwrite=args.overwrite)

    print(f"[OK] sample -> {args.out_jsonl}")
    if not args.no_tsv:
        print(f"[OK] sample_tsv -> {args.out_jsonl.with_suffix('.tsv')}")


if __name__ == "__main__":
    main()
