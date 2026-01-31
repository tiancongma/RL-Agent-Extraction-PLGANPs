#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_manifest_to_text.py

Purpose
- Run the unified cleaner (pdf2clean.py, manifest mode) on the authoritative manifest TSV.
- Produce cleaned text files and the authoritative key2txt.tsv in data/cleaned/index/.

Design constraints
- No hard-coded repository paths. Default inputs/outputs must come from src/utils/paths.py.
- This script is a thin orchestrator. Actual parsing/cleaning stays in pdf2clean.py.

Inputs (defaults via paths.py)
- manifest: data/cleaned/index/manifest_current.tsv

Outputs (authoritative)
- cleaned text files: data/cleaned/content/text/
- key2txt.tsv:        data/cleaned/index/key2txt.tsv   (2-column mapping: key -> repo-root-relative text path)

Notes about pdf2clean.py behavior
- pdf2clean.py writes cleaned texts under: <out_dir>/text/
- and writes key2txt.tsv to: <out_dir>/key2txt.tsv

We therefore set out_dir = data/cleaned/content to align texts with:
  data/cleaned/content/text

Important:
- pdf2clean's produced key2txt.tsv is a richer status table (header + many columns) and
  its txt_path values are relative to <out_dir> (e.g., "text\\XXXX.pdf.txt").
- Downstream stages require a stable 2-column mapping file.
- This script promotes a normalized 2-column mapping into data/cleaned/index/key2txt.tsv.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "src" / "utils" / "paths.py").exists():
            sys.path.insert(0, str(p))
            return


_bootstrap_import_paths()
from src.utils import paths  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Clean manifest -> text + key2txt.tsv (authoritative mapping).")

    ap.add_argument(
        "--manifest",
        type=Path,
        default=(paths.DATA_CLEANED_INDEX_DIR / "manifest_current.tsv"),
        help="Input manifest TSV. Default: data/cleaned/index/manifest_current.tsv (via paths.py).",
    )

    ap.add_argument(
        "--prefer",
        choices=["html", "pdf"],
        default="html",
        help="Prefer HTML or PDF when both exist. Default: html.",
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing cleaned text outputs (passed to pdf2clean).",
    )

    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging (passed to pdf2clean).",
    )

    ap.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Optional limit on PDF pages to parse (0 = no limit).",
    )

    return ap


def run_pdf2clean(manifest_path: Path, out_dir: Path, prefer: str, overwrite: bool, verbose: bool, max_pages: int) -> Path:
    """
    Invoke pdf2clean.py in manifest mode as a subprocess.
    Returns the path to the key2txt.tsv produced by pdf2clean under out_dir.
    """
    pdf2clean_path = paths.SRC_DIR / "stage1_cleaning" / "pdf2clean.py"
    if not pdf2clean_path.exists():
        raise FileNotFoundError(f"pdf2clean.py not found at expected location: {pdf2clean_path}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(pdf2clean_path),
        "--manifest", str(manifest_path),
        "--out-dir", str(out_dir),
        "--prefer", prefer,
    ]

    if overwrite:
        cmd.append("--overwrite")
    if verbose:
        cmd.append("--verbose")
    if max_pages and max_pages > 0:
        cmd += ["--max-pages", str(max_pages)]

    # Ensure expected folders exist (even if pdf2clean doesn't create all of them)
    (out_dir / "text").mkdir(parents=True, exist_ok=True)
    # Keep these as stable subdirs under content even if currently unused
    (out_dir / "sections").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
        raise RuntimeError(f"pdf2clean failed with return code {proc.returncode}")

    produced_key2 = out_dir / "key2txt.tsv"
    if not produced_key2.exists():
        raise FileNotFoundError(f"pdf2clean did not produce key2txt.tsv at: {produced_key2}")

    return produced_key2


def promote_key2txt_mapping(
    produced_key2txt: Path,
    content_dir: Path,
    dest_key2txt: Path,
    overwrite: bool,
) -> None:
    """
    Promote a normalized 2-column mapping into data/cleaned/index/key2txt.tsv.

    Input: pdf2clean-produced key2txt.tsv (header + many cols, includes txt_path relative to content_dir)
    Output: dest_key2txt with 2 columns, no header:
        <key>\t<repo-root-relative-text-path>

    Only rows whose resolved file exists on disk are included.
    """
    dest_key2txt.parent.mkdir(parents=True, exist_ok=True)
    if dest_key2txt.exists() and not overwrite:
        raise FileExistsError(f"Destination key2txt exists (use --overwrite): {dest_key2txt}")

    repo_root = paths.PROJECT_ROOT

    def to_repo_rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(repo_root.resolve()))
        except Exception:
            # fall back to absolute if outside repo root
            return str(p.resolve())

    rows_written = 0
    with produced_key2txt.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # Expect at least these columns from pdf2clean: key, txt_path
        if not reader.fieldnames or ("key" not in reader.fieldnames) or ("txt_path" not in reader.fieldnames):
            raise ValueError(
                f"Unexpected key2txt schema in {produced_key2txt}. "
                f"Expected tab-delimited header containing at least 'key' and 'txt_path'."
            )

        with dest_key2txt.open("w", encoding="utf-8", newline="") as out:
            for rec in reader:
                k = (rec.get("key") or "").strip()
                rel = (rec.get("txt_path") or "").strip()
                if not k or not rel:
                    continue
                full = (content_dir / rel).resolve()
                if not full.exists():
                    continue
                out.write(f"{k}\t{to_repo_rel(full)}\n")
                rows_written += 1

    if rows_written == 0:
        raise RuntimeError(
            "Promoted key2txt mapping has 0 rows. "
            "This indicates txt_path values could not be resolved to existing files."
        )


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    # Align with frozen structure from paths.py:
    # cleaned content under data/cleaned/content/
    content_dir = paths.DATA_CLEANED_CONTENT_DIR

    produced_key2 = run_pdf2clean(
        manifest_path=args.manifest,
        out_dir=content_dir,
        prefer=args.prefer,
        overwrite=args.overwrite,
        verbose=args.verbose,
        max_pages=args.max_pages,
    )

    dest_key2 = paths.DATA_CLEANED_INDEX_DIR / "key2txt.tsv"
    promote_key2txt_mapping(
        produced_key2txt=produced_key2,
        content_dir=content_dir,
        dest_key2txt=dest_key2,
        overwrite=True,
    )

    print(f"[OK] cleaned text dir: {content_dir / 'text'}")
    print(f"[OK] key2txt promoted (2-col mapping) -> {dest_key2}")


if __name__ == "__main__":
    main()
