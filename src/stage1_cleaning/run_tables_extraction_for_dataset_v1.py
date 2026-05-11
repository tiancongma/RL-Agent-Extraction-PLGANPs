#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
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


def _read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"manifest has no header: {path}")
        rows = [{k: (v or "") for k, v in row.items()} for row in reader]
        headers = list(reader.fieldnames)
    if "zotero_key" in headers:
        key_col = "zotero_key"
    elif "key" in headers:
        key_col = "key"
    else:
        raise ValueError("manifest must contain 'zotero_key' or 'key' column")
    return headers, rows, key_col


def _first_nonempty(row: dict[str, str], *cands: str) -> str:
    for c in cands:
        v = (row.get(c) or "").strip()
        if v:
            return v
    return ""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run Stage1 table extraction for all keys in a dataset manifest (HTML preferred, PDF fallback)."
    )
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--manifest-tsv", default="")
    ap.add_argument("--tables-root", default="")
    ap.add_argument("--keys-file", default="", help="Optional key list (one per line or TSV with zotero_key column).")
    ap.add_argument("--coverage-out", default="", help="Optional output coverage TSV path.")
    ap.add_argument("--n", type=int, default=0, help="Optional limit on number of keys (0 = all).")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _read_keys_file(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    if not lines:
        return []
    first = lines[0]
    if "\t" in first and "zotero_key" in first.split("\t"):
        idx = first.split("\t").index("zotero_key")
        out: list[str] = []
        for ln in lines[1:]:
            parts = ln.split("\t")
            if idx < len(parts):
                k = parts[idx].strip()
                if k:
                    out.append(k)
        return out
    return lines


def main() -> int:
    args = parse_args()
    dataset_id = args.dataset_id.strip()
    if not dataset_id:
        raise ValueError("dataset-id is empty")

    dataset_root = paths.DATA_CLEANED_DIR / dataset_id
    manifest_tsv = Path(args.manifest_tsv) if args.manifest_tsv else dataset_root / "index" / "manifest.tsv"
    tables_root = Path(args.tables_root) if args.tables_root else dataset_root / "tables"
    analysis_dir = dataset_root / "analysis"
    tables_index_path = dataset_root / "tables_index.tsv"

    if not manifest_tsv.exists():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_tsv}")
    tables_root.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    headers, rows, key_col = _read_manifest(manifest_tsv)
    by_key: dict[str, dict[str, str]] = {}
    for r in rows:
        k = (r.get(key_col) or "").strip()
        if not k:
            continue
        if k not in by_key:
            by_key[k] = r
    keys = sorted(by_key.keys())
    if args.keys_file:
        user_keys_path = Path(args.keys_file)
        if not user_keys_path.exists():
            raise FileNotFoundError(f"keys-file not found: {user_keys_path}")
        wanted = sorted(dict.fromkeys(_read_keys_file(user_keys_path)))
        keys = [k for k in wanted if k in by_key]

    if args.n > 0 and args.n < len(keys):
        rnd = random.Random(args.seed)
        keys = sorted(rnd.sample(keys, args.n))

    keys_file = analysis_dir / "tables_extraction_keys_v1.txt"
    keys_file.write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")

    extractor = paths.SRC_DIR / "stage1_cleaning" / "extract_tables_for_keys_v1.py"
    cmd = [
        sys.executable,
        str(extractor),
        "--keys-file",
        str(keys_file),
        "--tables-root",
        str(tables_root),
        "--tables-index-path",
        str(tables_index_path),
        "--skip-post-check",
    ]
    print(f"run_cmd={' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.returncode != 0:
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"table extractor failed with return code {proc.returncode}")

    coverage_path = Path(args.coverage_out) if args.coverage_out else analysis_dir / "tables_extraction_coverage.tsv"
    out_rows: list[dict[str, str]] = []
    for k in keys:
        row = by_key.get(k, {})
        doi = _first_nonempty(row, "doi")
        html_path = _first_nonempty(row, "html")
        pdf_path = _first_nonempty(row, "pdf")
        html_found = "True" if html_path else "False"
        pdf_found = "True" if pdf_path else "False"
        km = tables_root / k / "tables_manifest.json"
        n_html = 0
        n_pdf = 0
        html_reason = ""
        pdf_reason = ""
        chosen_source = "none"
        if km.exists():
            import json

            obj = json.loads(km.read_text(encoding="utf-8", errors="replace"))
            n_html = int(obj.get("n_tables_html_extracted", 0) or 0)
            n_pdf = int(obj.get("n_tables_pdf_extracted", 0) or 0)
            html_reason = str(obj.get("html_table_reason", "") or "")
            pdf_reason = str(obj.get("pdf_table_reason", "") or "")
            chosen_source = str(obj.get("preferred_table_source", "none") or "none")
            html_found = "True" if bool(obj.get("html_found", False)) else "False"
            pdf_found = "True" if bool(obj.get("pdf_found", False)) else "False"
        if pdf_found == "True" and n_pdf == 0 and not pdf_reason.strip():
            pdf_reason = "no_tables_detected"
        out_rows.append(
            {
                "zotero_key": k,
                "doi": doi,
                "html_found": html_found,
                "pdf_found": pdf_found,
                "n_tables_html_extracted": str(n_html),
                "n_tables_pdf_extracted": str(n_pdf),
                "chosen_source": chosen_source,
                "html_table_reason": html_reason,
                "pdf_table_reason": pdf_reason,
            }
        )

    with coverage_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "zotero_key",
            "doi",
            "html_found",
            "pdf_found",
            "n_tables_html_extracted",
            "n_tables_pdf_extracted",
            "chosen_source",
            "html_table_reason",
            "pdf_table_reason",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)

    n_total = len(out_rows)
    n_html_found = sum(1 for r in out_rows if r["html_found"] == "True")
    n_pdf_found = sum(1 for r in out_rows if r["pdf_found"] == "True")
    n_any_tables = sum(
        1
        for r in out_rows
        if int(r["n_tables_html_extracted"]) + int(r["n_tables_pdf_extracted"]) > 0
    )
    n_chosen_html = sum(1 for r in out_rows if r["chosen_source"] == "html")
    n_chosen_pdf = sum(1 for r in out_rows if r["chosen_source"] == "pdf")
    print("tables_extraction_dataset_summary")
    print(f"dataset_id\t{dataset_id}")
    print(f"n_total_keys\t{n_total}")
    print(f"n_html_found_true\t{n_html_found}")
    print(f"n_pdf_found_true\t{n_pdf_found}")
    print(f"n_any_tables_extracted\t{n_any_tables}")
    print(f"n_chosen_source_html\t{n_chosen_html}")
    print(f"n_chosen_source_pdf\t{n_chosen_pdf}")
    print(f"coverage_tsv\t{coverage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
