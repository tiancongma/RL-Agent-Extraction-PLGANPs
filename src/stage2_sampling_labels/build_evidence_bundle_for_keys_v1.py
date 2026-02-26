#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import paths as P
from src.utils.run_id import is_valid_run_id
from src.utils.run_latest import inputs_fingerprint, write_latest


def _read_keys(keys_file: Path) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in keys_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        key = raw.strip()
        if not key or key.startswith("#"):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _pick_col(df: pd.DataFrame, preferred: str) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    return cols.get(preferred.lower())


def _manifest_maps(manifest_path: Path) -> Tuple[str, Dict[str, str]]:
    df = pd.read_csv(manifest_path, sep="\t", dtype=str, keep_default_na=False).fillna("")
    key_col = _pick_col(df, "key") or df.columns[0]
    doi_col = _pick_col(df, "doi")
    doi_map: Dict[str, str] = {}
    for _, row in df.iterrows():
        k = str(row.get(key_col, "")).strip()
        if not k:
            continue
        doi = str(row.get(doi_col, "")).strip() if doi_col else ""
        if k not in doi_map:
            doi_map[k] = doi
    return key_col, doi_map


def _read_text_for_key(content_dir: Path, key: str) -> Tuple[str, str, bool]:
    txt_dir = content_dir / "text"
    html_txt = txt_dir / f"{key}.html.txt"
    pdf_txt = txt_dir / f"{key}.pdf.txt"
    for cand in (html_txt, pdf_txt):
        if cand.exists():
            return str(cand), cand.read_text(encoding="utf-8", errors="ignore"), False
    return "", "", True


def _df_to_tsv_str(df: pd.DataFrame) -> str:
    clean = df.fillna("").astype(str)
    buf = io.StringIO()
    clean.to_csv(buf, sep="\t", index=False, lineterminator="\n")
    return buf.getvalue()


def _sanitize_out_subdir(s: str) -> str:
    v = str(s or "").strip().replace("\\", "/")
    if not v:
        raise ValueError(
            "ERROR: --out-subdir is required when reusing a run_id. Use a stage/variant folder name, e.g. stage2_validation or stage5_signature_iter001."
        )
    if Path(v).is_absolute():
        raise ValueError("ERROR: --out-subdir must be a relative path.")
    parts = [p for p in v.split("/") if p]
    if not parts or any(p == ".." for p in parts):
        raise ValueError("ERROR: --out-subdir cannot contain path traversal ('..').")
    return "/".join(parts)


def build_evidence(
    manifest_path: Path,
    keys_file: Path,
    content_dir: Path,
    run_id: str,
    out_jsonl: Path | None,
    out_subdir: str,
) -> Tuple[Path, Path, int]:
    if not is_valid_run_id(str(run_id).strip()):
        raise ValueError(f"Invalid run_id (must match required regex): {run_id}")
    _, doi_map = _manifest_maps(manifest_path)
    keys = _read_keys(keys_file)

    out_base = P.DATA_RESULTS_DIR / run_id
    if str(out_subdir).strip():
        out_base = out_base / str(out_subdir).strip()
    out_base = out_base / "stage2_validation"
    out_base.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_jsonl or (out_base / "evidence_bundles_v1.jsonl")
    summary_path = out_base / "evidence_bundles_v1_summary.tsv"

    records: List[dict] = []
    summary_rows: List[dict] = []

    for key in keys:
        notes: List[str] = []
        source_text_path, source_text, missing_text = _read_text_for_key(content_dir, key)
        if missing_text:
            notes.append("missing_text: no <key>.html.txt or <key>.pdf.txt")

        table_key_dir = content_dir / "tables" / key
        table_manifest_path = table_key_dir / "tables_manifest.json"
        missing_table_manifest = not table_manifest_path.exists()

        preferred_table_source = "none"
        selected_table_files: List[str] = []
        selected_tables_tsv: List[str] = []

        if missing_table_manifest:
            notes.append(f"missing_table_manifest: {table_manifest_path}")
        else:
            try:
                tm = json.loads(table_manifest_path.read_text(encoding="utf-8", errors="ignore"))
            except Exception as e:
                tm = {}
                notes.append(f"tables_manifest_json_parse_failed: {e}")

            preferred_table_source = str(tm.get("preferred_table_source", "none") or "none")
            raw_files = tm.get("selected_table_files", [])
            if isinstance(raw_files, list):
                selected_table_files = [Path(str(x)).name for x in raw_files if str(x).strip()]
            else:
                notes.append("selected_table_files_not_list")

            for fname in selected_table_files:
                fpath = table_key_dir / fname
                if not fpath.exists():
                    notes.append(f"missing_selected_table_file: {fpath}")
                    selected_tables_tsv.append("")
                    continue
                try:
                    tdf = pd.read_csv(fpath, dtype=str, keep_default_na=False)
                    selected_tables_tsv.append(_df_to_tsv_str(tdf))
                except Exception as e:
                    notes.append(f"selected_table_read_failed:{fname}: {e}")
                    selected_tables_tsv.append("")

        rec = {
            "key": key,
            "doi": doi_map.get(key, ""),
            "source_text_path": source_text_path,
            "source_text": source_text,
            "preferred_table_source": preferred_table_source,
            "selected_table_files": selected_table_files,
            "selected_tables_tsv": selected_tables_tsv,
            "notes": notes,
        }
        records.append(rec)

        summary_rows.append(
            {
                "key": key,
                "has_text": int(bool(source_text)),
                "n_selected_tables": len(selected_table_files),
                "preferred_table_source": preferred_table_source,
                "missing_table_manifest": int(missing_table_manifest),
                "missing_text": int(missing_text),
            }
        )

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
    return jsonl_path, summary_path, len(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build downstream evidence bundles (text + selected tables) for a list of keys."
    )
    parser.add_argument(
        "--manifest",
        default=str(P.DATA_CLEANED_INDEX_DIR / "manifest_goren_2025.tsv"),
        help="Path to manifest TSV.",
    )
    parser.add_argument("--keys-file", required=True, help="One key per line.")
    parser.add_argument(
        "--content-dir",
        default=str(P.DATA_CLEANED_DIR / "content_goren_2025"),
        help="Content root containing text/ and tables/.",
    )
    parser.add_argument("--run-id", default="", help="Required deterministic run_id from preflight.")
    parser.add_argument(
        "--out-subdir",
        default="",
        help="Optional subdirectory under data/results/<run_id>/ for run variants (e.g., iter_001).",
    )
    parser.add_argument(
        "--out-jsonl",
        default="",
        help="Optional explicit JSONL output path. Default: data/results/<run_id>/stage2_validation/evidence_bundles_v1.jsonl",
    )
    args = parser.parse_args()
    rid = str(args.run_id).strip()
    if not rid:
        raise ValueError(
            "ERROR: --run-id is required. Generate/reuse a run_id via: python -m src.utils.run_preflight ..."
        )
    if not is_valid_run_id(rid):
        raise ValueError(f"Invalid --run-id (must match required regex): {rid}")
    out_subdir = _sanitize_out_subdir(args.out_subdir)

    latest_path = write_latest(
        run_id=rid,
        meta={
            "subset": "goren2025",
            "stage": "stage2_validation",
            "inputs_fingerprint": inputs_fingerprint([Path(args.manifest), Path(args.keys_file)]),
            "note": "build_evidence_bundle_for_keys_v1",
        },
    )

    out_jsonl_arg = Path(args.out_jsonl) if str(args.out_jsonl).strip() else None
    if out_jsonl_arg is not None:
        base = (P.DATA_RESULTS_DIR / rid / out_subdir).resolve()
        try:
            out_jsonl_arg.resolve().relative_to(base)
        except Exception:
            raise ValueError(
                f"ERROR: --out-jsonl must be under data/results/<run_id>/<out-subdir>/. Got: {out_jsonl_arg}"
            )

    jsonl_path, summary_path, n = build_evidence(
        manifest_path=Path(args.manifest),
        keys_file=Path(args.keys_file),
        content_dir=Path(args.content_dir),
        run_id=rid,
        out_jsonl=out_jsonl_arg,
        out_subdir=out_subdir,
    )
    print(f"run_id\t{rid}")
    print(f"latest_pointer\t{latest_path}")
    print(f"records_written\t{n}")
    print(f"out_jsonl\t{jsonl_path}")
    print(f"out_summary_tsv\t{summary_path}")


if __name__ == "__main__":
    main()
