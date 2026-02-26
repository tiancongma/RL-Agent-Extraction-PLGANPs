from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REQUIRED_SUBDIRS = ("index", "content", "text", "sections", "tables", "analysis")


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"TSV has no header: {path}")
        rows = [{k: (v or "") for k, v in row.items()} for row in reader]
        return list(reader.fieldnames), rows


def _normalize_manifest(
    headers: list[str], rows: list[dict[str, str]]
) -> tuple[list[str], list[dict[str, str]], str]:
    if "zotero_key" in headers:
        key_col = "zotero_key"
        return headers, rows, key_col
    if "key" in headers:
        new_headers = ["zotero_key" if h == "key" else h for h in headers]
        out_rows: list[dict[str, str]] = []
        for row in rows:
            out_row = {}
            for h in headers:
                if h == "key":
                    out_row["zotero_key"] = row.get("key", "")
                else:
                    out_row[h] = row.get(h, "")
            out_rows.append(out_row)
        return new_headers, out_rows, "zotero_key"
    raise ValueError("manifest must contain 'zotero_key' or 'key' column")


def _write_manifest(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _safe_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if mode == "hardlink":
        try:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"
    if mode == "symlink":
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
            return "symlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"
    raise ValueError(f"unsupported mode: {mode}")


def _iter_files(path: Path) -> Iterable[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted((p for p in path.iterdir() if p.is_file()), key=lambda p: p.name)


def _write_readme(path: Path, content: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a dataset-scoped cleaned root from a legacy cleaned root."
    )
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--legacy-root", required=True, help="Legacy root, e.g. data/cleaned/content_goren_2025")
    ap.add_argument("--out-cleaned-root", default="data/cleaned")
    ap.add_argument("--manifest-tsv", required=True)
    ap.add_argument("--mode", choices=("copy", "hardlink", "symlink"), default="copy")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dataset_id = args.dataset_id.strip()
    if not dataset_id:
        raise ValueError("dataset-id is empty")
    if dataset_id.startswith("content_"):
        raise ValueError("dataset-id must be semantic-only and must not use content_* prefix")

    legacy_root = Path(args.legacy_root)
    manifest_src = Path(args.manifest_tsv)
    cleaned_root = Path(args.out_cleaned_root)
    dataset_root = cleaned_root / dataset_id

    headers, rows = _read_tsv(manifest_src)
    out_headers, out_rows, key_col = _normalize_manifest(headers, rows)
    keys = sorted({r.get(key_col, "").strip() for r in out_rows if r.get(key_col, "").strip()})

    if not args.dry_run:
        for d in REQUIRED_SUBDIRS:
            (dataset_root / d).mkdir(parents=True, exist_ok=True)
        _write_manifest(dataset_root / "index" / "manifest.tsv", out_headers, out_rows)

    legacy_tables_root = legacy_root / "tables"
    legacy_text_root = legacy_root / "text"
    legacy_sections_root = legacy_root / "sections"

    n_keys_tables_mapped = 0
    n_keys_tables_missing = 0
    missing_table_keys: list[str] = []
    n_text_files_mapped = 0
    n_sections_files_mapped = 0
    copy_action_counts: dict[str, int] = {
        "copy": 0,
        "hardlink": 0,
        "symlink": 0,
        "copy_fallback": 0,
    }

    for key in keys:
        legacy_key_tables = legacy_tables_root / key
        out_key_tables = dataset_root / "tables" / key
        if legacy_key_tables.exists() and legacy_key_tables.is_dir():
            n_keys_tables_mapped += 1
            for src in _iter_files(legacy_key_tables):
                if args.dry_run:
                    continue
                action = _safe_copy(src, out_key_tables / src.name, args.mode)
                copy_action_counts[action] = copy_action_counts.get(action, 0) + 1
        else:
            n_keys_tables_missing += 1
            missing_table_keys.append(key)

        # Map legacy text files to paper-local text/<key>/ if they exist.
        if legacy_text_root.exists() and legacy_text_root.is_dir():
            matches = sorted(
                (
                    p
                    for p in _iter_files(legacy_text_root)
                    if p.name == key or p.name.startswith(f"{key}.")
                ),
                key=lambda p: p.name,
            )
            for src in matches:
                if args.dry_run:
                    continue
                action = _safe_copy(src, dataset_root / "text" / key / src.name, args.mode)
                copy_action_counts[action] = copy_action_counts.get(action, 0) + 1
                n_text_files_mapped += 1

        # Map legacy sections files to paper-local sections/<key>/ if they exist.
        if legacy_sections_root.exists() and legacy_sections_root.is_dir():
            matches = sorted(
                (
                    p
                    for p in _iter_files(legacy_sections_root)
                    if p.name == key or p.name.startswith(f"{key}.")
                ),
                key=lambda p: p.name,
            )
            for src in matches:
                if args.dry_run:
                    continue
                action = _safe_copy(src, dataset_root / "sections" / key / src.name, args.mode)
                copy_action_counts[action] = copy_action_counts.get(action, 0) + 1
                n_sections_files_mapped += 1

    # Leave placeholders when the legacy source has no mapped assets.
    if not args.dry_run:
        if n_text_files_mapped == 0:
            _write_readme(
                dataset_root / "text" / "README.md",
                "No per-key text artifacts were mapped from legacy root yet.\n",
                dry_run=False,
            )
        if n_sections_files_mapped == 0:
            _write_readme(
                dataset_root / "sections" / "README.md",
                "No per-key sections artifacts were mapped from legacy root yet.\n",
                dry_run=False,
            )

    report = {
        "dataset_id": dataset_id,
        "n_keys_total": len(keys),
        "n_keys_tables_mapped": n_keys_tables_mapped,
        "n_keys_tables_missing": n_keys_tables_missing,
        "mode_used": args.mode,
        "legacy_root": str(legacy_root),
        "manifest_source_path": str(manifest_src),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_text_files_mapped": n_text_files_mapped,
        "n_sections_files_mapped": n_sections_files_mapped,
        "copy_action_counts": copy_action_counts,
        "missing_table_keys": missing_table_keys,
    }

    if not args.dry_run:
        report_path = dataset_root / "analysis" / "dataset_build_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("build_dataset_root_from_legacy_v1")
    print(f"dataset_root\t{dataset_root}")
    print(f"manifest_source\t{manifest_src}")
    print(f"mode\t{args.mode}")
    print(f"dry_run\t{args.dry_run}")
    print(f"n_keys_total\t{report['n_keys_total']}")
    print(f"n_keys_tables_mapped\t{report['n_keys_tables_mapped']}")
    print(f"n_keys_tables_missing\t{report['n_keys_tables_missing']}")
    print(f"n_text_files_mapped\t{report['n_text_files_mapped']}")
    print(f"n_sections_files_mapped\t{report['n_sections_files_mapped']}")
    if report["missing_table_keys"]:
        preview = ",".join(report["missing_table_keys"][:20])
        print(f"missing_table_keys_first20\t{preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
