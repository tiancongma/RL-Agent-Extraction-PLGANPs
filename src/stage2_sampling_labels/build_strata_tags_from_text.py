#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_strata_tags_from_text.py

Purpose
- Deterministically generate lightweight, rule-based strata tags from cleaned text,
  to enable reproducible stratified sampling (e.g., sample20) without LLM calls.

Inputs (defaults via paths.py)
- key2txt mapping (authoritative, 2-column): data/cleaned/index/key2txt.tsv
- optional cleaning summary table:            data/cleaned/content/key2txt.tsv

Outputs
- strata_tags.tsv: data/cleaned/index/strata_tags.tsv

Tag definitions (best-effort, intentionally coarse)
- particle_scale_tag: nano | micro | mixed | unknown
- emulsion_route_tag: O/W | W/O/W | mixed | unknown
- reporting_style_tag: table | text | unknown   (proxy via table_detected when available)
- source_type: HTML if any HTML exists for the key in cleaning summary; else PDF if any PDF exists; else blank

Determinism
- No randomness.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Optional

# ---- bootstrap repo root for "import src.*" when running as a script ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths  # noqa: E402


# Regexes (coarse and conservative)
_NANO_RE = re.compile(r"\b(nano(?:particle|sphere|emulsion|suspension)?s?)\b", re.IGNORECASE)
_MICRO_RE = re.compile(r"\b(micro(?:particle|sphere|emulsion|suspension)?s?|microsphere|microparticle)\b", re.IGNORECASE)

_WOW_RE = re.compile(r"\b(w\/o\/w|w1\/o\/w2|w1\s*\/\s*o\s*\/\s*w2|double\s+emulsion)\b", re.IGNORECASE)
_OW_RE = re.compile(r"\b(o\/w|oil\s*\/\s*water|single\s+emulsion)\b", re.IGNORECASE)


def _read_key2txt_map(map_path: Path) -> Dict[str, Path]:
    """
    Read 2-column mapping file: key <tab> path (repo-relative or absolute).
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


def _coerce_bool(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    t = str(s).strip().lower()
    if t in {"1", "true", "t", "yes", "y"}:
        return True
    if t in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _safe_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    try:
        return int(float(t))
    except Exception:
        return None


def _infer_particle_scale(text: str) -> str:
    nano = bool(_NANO_RE.search(text))
    micro = bool(_MICRO_RE.search(text))
    if nano and not micro:
        return "nano"
    if micro and not nano:
        return "micro"
    if nano and micro:
        return "mixed"
    return "unknown"


def _infer_emulsion_route(text: str) -> str:
    wow = bool(_WOW_RE.search(text))
    ow = bool(_OW_RE.search(text))
    if wow and not ow:
        return "W/O/W"
    if ow and not wow:
        return "O/W"
    if wow and ow:
        return "mixed"
    return "unknown"


def _infer_reporting_style(table_detected: Optional[bool]) -> str:
    if table_detected is True:
        return "table"
    if table_detected is False:
        return "text"
    return "unknown"


def _safe_read_text(p: Path, max_chars: int) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


def _read_cleaning_summary_agg(summary_path: Path) -> Dict[str, dict]:
    """
    Aggregate pdf2clean-produced summary table (header TSV) at data/cleaned/content/key2txt.tsv.

    For each key, aggregate:
    - source_type: HTML if any HTML exists for the key; else PDF if any PDF exists; else ""
    - table_detected: True if any row True; else False if any row False; else None
    - text_length: max numeric value across rows (if available)
    - parse_quality: first non-empty
    - notes: first non-empty
    """
    meta: Dict[str, dict] = {}
    if not summary_path.exists():
        return meta

    with summary_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or "key" not in reader.fieldnames:
            return meta

        for rec in reader:
            k = (rec.get("key") or "").strip()
            if not k:
                continue

            st = (rec.get("source_type") or "").strip().upper()
            td = _coerce_bool(rec.get("table_detected"))
            tl = _safe_int(rec.get("text_length"))
            pq = (rec.get("parse_quality") or "").strip()
            notes = (rec.get("notes") or "").strip()

            if k not in meta:
                meta[k] = {
                    "source_types": set(),
                    "table_detected_any_true": False,
                    "table_detected_any_false": False,
                    "text_length_max": None,
                    "parse_quality": "",
                    "notes": "",
                }

            if st:
                meta[k]["source_types"].add(st)

            if td is True:
                meta[k]["table_detected_any_true"] = True
            elif td is False:
                meta[k]["table_detected_any_false"] = True

            if tl is not None:
                cur = meta[k]["text_length_max"]
                meta[k]["text_length_max"] = tl if (cur is None or tl > cur) else cur

            if pq and not meta[k]["parse_quality"]:
                meta[k]["parse_quality"] = pq
            if notes and not meta[k]["notes"]:
                meta[k]["notes"] = notes

    # finalize
    out: Dict[str, dict] = {}
    for k, v in meta.items():
        sts = v["source_types"]
        if "HTML" in sts:
            src = "HTML"
        elif "PDF" in sts:
            src = "PDF"
        else:
            src = ""

        if v["table_detected_any_true"]:
            td_final = True
        elif v["table_detected_any_false"]:
            td_final = False
        else:
            td_final = None

        out[k] = {
            "source_type": src,
            "table_detected": td_final,
            "text_length": v["text_length_max"],
            "parse_quality": v["parse_quality"],
            "notes": v["notes"],
        }
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build strata tag table (rule-based) from cleaned texts.")
    ap.add_argument(
        "--key2txt",
        type=Path,
        default=(paths.DATA_CLEANED_INDEX_DIR / "key2txt.tsv"),
        help="Authoritative 2-col key2txt mapping (default: data/cleaned/index/key2txt.tsv).",
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=(paths.DATA_CLEANED_CONTENT_DIR / "key2txt.tsv"),
        help="Optional cleaning summary table (default: data/cleaned/content/key2txt.tsv).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=(paths.DATA_CLEANED_INDEX_DIR / "strata_tags.tsv"),
        help="Output TSV (default: data/cleaned/index/strata_tags.tsv).",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=120000,
        help="Max chars to scan from each cleaned text (default: 120000).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if exists.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.key2txt.exists():
        raise SystemExit(f"key2txt mapping not found: {args.key2txt}")

    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite (use --overwrite): {args.out}")

    key2txt = _read_key2txt_map(args.key2txt)
    summary = _read_cleaning_summary_agg(args.summary)

    out_rows = []
    missing_files = 0

    for k, p in key2txt.items():
        if not p.exists():
            missing_files += 1
            continue

        meta = summary.get(k, {})
        src_type = (meta.get("source_type") or "").strip()
        parse_quality = (meta.get("parse_quality") or "").strip()
        notes = (meta.get("notes") or "").strip()

        td = meta.get("table_detected")
        reporting_style = _infer_reporting_style(td if isinstance(td, bool) else None)

        text_length = meta.get("text_length")
        if isinstance(text_length, int):
            tl_out = str(text_length)
        else:
            tl_out = ""

        txt = _safe_read_text(p, max_chars=args.max_chars)
        particle = _infer_particle_scale(txt)
        emulsion = _infer_emulsion_route(txt)

        # If summary didn't provide text_length, compute from scanned text length as fallback
        if not tl_out:
            tl_out = str(len(txt)) if txt else ""

        out_rows.append({
            "key": k,
            "particle_scale_tag": particle,
            "emulsion_route_tag": emulsion,
            "reporting_style_tag": reporting_style,
            "source_type": src_type,
            "text_length": tl_out,
            "table_detected": "" if td is None else ("1" if td else "0"),
            "parse_quality": parse_quality,
            "notes": notes,
            "text_path": str(p.resolve()),
        })

    if args.verbose:
        print(f"[info] keys in mapping: {len(key2txt)}")
        print(f"[info] missing text files on disk: {missing_files}")
        print(f"[info] tags produced: {len(out_rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "key",
            "particle_scale_tag",
            "emulsion_route_tag",
            "reporting_style_tag",
            "source_type",
            "text_length",
            "table_detected",
            "parse_quality",
            "notes",
            "text_path",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"[OK] strata_tags -> {args.out}")


if __name__ == "__main__":
    main()
