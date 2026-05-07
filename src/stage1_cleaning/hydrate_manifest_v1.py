#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hydrate an assembled Stage1 manifest into the fully usable canonical manifest.

Fine-grained Stage1 contract:
- S1-2: multi-source manifest assembly
- S1-3: manifest hydration
  - S1-3a: asset hydration
  - S1-3b: scope overlays

This helper is deterministic. It does not rerun cleaning or table extraction.
It binds an already-assembled manifest to explicit, already-governed Stage1
surfaces.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "src" / "utils" / "paths.py").exists():
            sys.path.insert(0, str(p))
            return


_bootstrap_import_paths()
from src.utils import paths  # noqa: E402


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_key(value: Any) -> str:
    return normalize_text(value)


def to_repo_rel_windows(path_value: Path) -> str:
    try:
        rel = path_value.resolve().relative_to(paths.PROJECT_ROOT.resolve())
        return str(rel).replace("/", "\\")
    except Exception:
        return str(path_value.resolve())


def resolve_project_path(path_value: Path | str) -> Path:
    path_obj = path_value if isinstance(path_value, Path) else Path(path_value)
    return path_obj if path_obj.is_absolute() else (paths.PROJECT_ROOT / path_obj).resolve()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_key2txt_map(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"key2txt surface not found: {path}")
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.readline()
        handle.seek(0)
        first_cells = sample.rstrip("\n\r").split("\t")
        if first_cells and first_cells[0] in {"key", "paper_key", "zotero_key"}:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                key = normalize_key(row.get("key") or row.get("paper_key") or row.get("zotero_key"))
                text_path = normalize_text(row.get("txt_path") or row.get("text_path") or row.get("path"))
                if key and text_path:
                    out[key] = text_path.replace("/", "\\")
        else:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue
                key = normalize_key(row[0])
                text_path = normalize_text(row[1])
                if key and text_path:
                    out[key] = text_path.replace("/", "\\")
    return out


def load_key2structure_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"key2structure surface not found: {path}")
    _, rows = read_tsv(path)
    out: dict[str, str] = {}
    for row in rows:
        key = normalize_key(row.get("key") or row.get("paper_key") or row.get("zotero_key"))
        structure_path = normalize_text(row.get("structure_path") or row.get("stage1_structure_path"))
        if key and structure_path:
            out[key] = structure_path.replace("\\", "/")
    return out


def load_stage1_table_cell_sidecar_map(path: Path | None) -> dict[str, tuple[str, str]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Stage1 table-cell sidecar manifest not found: {path}")
    _, rows = read_tsv(path)
    out: dict[str, tuple[str, str]] = {}
    for row in rows:
        key = normalize_key(row.get("key") or row.get("paper_key") or row.get("zotero_key"))
        sidecar_path = normalize_text(
            row.get("stage1_table_cell_sidecar_path")
            or row.get("table_cell_sidecar_path")
            or row.get("sidecar_path")
            or row.get("stage1_cells_path")
        )
        available = normalize_text(row.get("stage1_table_cell_sidecar_available") or row.get("available"))
        if key and sidecar_path:
            out[key] = (sidecar_path.replace("\\", "/"), available or "yes")
    return out


def extract_manifest_keys(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"dataset/split manifest not found: {path}")
    _, rows = read_tsv(path)
    out: set[str] = set()
    for row in rows:
        key = normalize_key(row.get("key") or row.get("zotero_key") or row.get("paper_key"))
        if key:
            out.add(key)
    return out


def infer_text_source_type(text_path: str) -> str:
    lower = normalize_text(text_path).lower()
    if lower.endswith(".html.txt"):
        return "html"
    if lower.endswith(".pdf.txt"):
        return "pdf"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-tsv",
        type=Path,
        required=True,
        help="Input assembled manifest TSV to hydrate.",
    )
    parser.add_argument(
        "--out-tsv",
        type=Path,
        required=True,
        help="Output hydrated manifest TSV.",
    )
    parser.add_argument(
        "--key2txt-tsv",
        type=Path,
        default=paths.DATA_CLEANED_INDEX_DIR / "key2txt.tsv",
        help="Authoritative key2txt mapping used for text hydration.",
    )
    parser.add_argument(
        "--key2structure-tsv",
        type=Path,
        default=paths.DATA_CLEANED_INDEX_DIR / "key2structure.tsv",
        help="Optional authoritative key2structure mapping used for structure sidecar hydration.",
    )
    parser.add_argument(
        "--table-cell-sidecar-manifest-tsv",
        type=Path,
        default=None,
        help="Optional Stage1 table-cell sidecar manifest used to hydrate per-paper sidecar paths.",
    )
    parser.add_argument(
        "--dataset-manifest-tsv",
        action="append",
        dest="dataset_manifests",
        default=[],
        help="Repeatable dataset manifest used for dataset overlay membership.",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        dest="dataset_ids",
        default=[],
        help="Repeatable dataset_id aligned to --dataset-manifest-tsv.",
    )
    parser.add_argument(
        "--dataset-tables-root",
        action="append",
        dest="dataset_tables_roots",
        default=[],
        help="Repeatable tables root aligned to --dataset-manifest-tsv. Default: data/cleaned/<dataset_id>/tables",
    )
    parser.add_argument(
        "--split-manifest-tsv",
        action="append",
        dest="split_manifests",
        default=[],
        help="Repeatable split manifest used for scope overlays.",
    )
    parser.add_argument(
        "--split-tag",
        action="append",
        dest="split_tags",
        default=[],
        help="Repeatable split_tag aligned to --split-manifest-tsv.",
    )
    parser.add_argument(
        "--benchmark-tag",
        action="append",
        dest="benchmark_tags",
        default=[],
        help="Repeatable benchmark_tag aligned to --split-manifest-tsv.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional hydration metadata sidecar path.",
    )
    parser.add_argument(
        "--strict-overlays",
        action="store_true",
        help="Fail if dataset or split overlays conflict for any key.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting --out-tsv if it already exists.",
    )
    return parser.parse_args()


def validate_aligned_args(values: list[str], arg_name: str, expected_len: int) -> list[str]:
    if len(values) != expected_len:
        raise ValueError(
            f"{arg_name} must be supplied exactly once per aligned manifest input "
            f"(expected {expected_len}, got {len(values)})"
        )
    return [normalize_text(value) for value in values]


def hydrate_manifest(
    *,
    manifest_tsv: Path,
    out_tsv: Path,
    key2txt_tsv: Path,
    key2structure_tsv: Path | None = None,
    table_cell_sidecar_manifest_tsv: Path | None = None,
    dataset_manifests: list[Path] | None = None,
    dataset_ids: list[str] | None = None,
    dataset_tables_roots: list[str] | None = None,
    split_manifests: list[Path] | None = None,
    split_tags: list[str] | None = None,
    benchmark_tags: list[str] | None = None,
    metadata_json: Path | None = None,
    strict_overlays: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_tsv = resolve_project_path(manifest_tsv)
    out_tsv = resolve_project_path(out_tsv)
    key2txt_tsv = resolve_project_path(key2txt_tsv)
    key2structure_resolved = resolve_project_path(key2structure_tsv) if key2structure_tsv is not None else None
    table_cell_sidecar_manifest_resolved = (
        resolve_project_path(table_cell_sidecar_manifest_tsv) if table_cell_sidecar_manifest_tsv is not None else None
    )

    if out_tsv.exists() and not overwrite:
        raise FileExistsError(f"output exists (use --overwrite): {out_tsv}")
    if not manifest_tsv.exists():
        raise FileNotFoundError(f"assembled manifest not found: {manifest_tsv}")

    dataset_manifests = [resolve_project_path(path) for path in (dataset_manifests or [])]
    dataset_ids = validate_aligned_args(dataset_ids or [], "--dataset-id", len(dataset_manifests))
    dataset_tables_roots = dataset_tables_roots or []
    if dataset_tables_roots and len(dataset_tables_roots) != len(dataset_manifests):
        raise ValueError(
            "--dataset-tables-root must be supplied zero times or exactly once per --dataset-manifest-tsv"
        )

    split_manifests = [resolve_project_path(path) for path in (split_manifests or [])]
    split_tags = validate_aligned_args(split_tags or [], "--split-tag", len(split_manifests))
    benchmark_tags = validate_aligned_args(benchmark_tags or [], "--benchmark-tag", len(split_manifests))

    fieldnames, rows = read_tsv(manifest_tsv)

    key2txt_map = load_key2txt_map(key2txt_tsv)
    key2structure_map = load_key2structure_map(key2structure_resolved)
    table_cell_sidecar_map = load_stage1_table_cell_sidecar_map(table_cell_sidecar_manifest_resolved)

    dataset_membership: dict[str, str] = {}
    dataset_tables_map: dict[str, Path] = {}
    dataset_conflicts: list[dict[str, str]] = []
    for idx, dataset_manifest in enumerate(dataset_manifests):
        dataset_id = dataset_ids[idx]
        if not dataset_id:
            raise ValueError(f"dataset_id is empty for dataset manifest: {dataset_manifest}")
        if dataset_tables_roots:
            tables_root = resolve_project_path(dataset_tables_roots[idx])
        else:
            tables_root = paths.dataset_tables_root(dataset_id)
        if not tables_root.exists():
            raise FileNotFoundError(f"declared tables root not found: {tables_root}")
        dataset_tables_map[dataset_id] = tables_root
        for key in extract_manifest_keys(dataset_manifest):
            previous = dataset_membership.get(key)
            if previous and previous != dataset_id:
                dataset_conflicts.append({"key": key, "previous_dataset_id": previous, "dataset_id": dataset_id})
            dataset_membership[key] = dataset_id

    split_overlay: dict[str, tuple[str, str]] = {}
    split_conflicts: list[dict[str, str]] = []
    for idx, split_manifest in enumerate(split_manifests):
        split_tag = split_tags[idx]
        benchmark_tag = benchmark_tags[idx]
        if not split_tag:
            raise ValueError(f"split_tag is empty for split manifest: {split_manifest}")
        for key in extract_manifest_keys(split_manifest):
            previous = split_overlay.get(key)
            current = (split_tag, benchmark_tag)
            if previous and previous != current:
                split_conflicts.append(
                    {
                        "key": key,
                        "previous_split_tag": previous[0],
                        "previous_benchmark_tag": previous[1],
                        "split_tag": split_tag,
                        "benchmark_tag": benchmark_tag,
                    }
                )
            split_overlay[key] = current

    if strict_overlays and (dataset_conflicts or split_conflicts):
        raise RuntimeError(
            f"overlay conflicts detected: dataset_conflicts={len(dataset_conflicts)} split_conflicts={len(split_conflicts)}"
        )

    hydrated_rows: list[dict[str, str]] = []
    hydration_summary: dict[str, Any] = {
        "input_manifest_path": str(manifest_tsv),
        "output_manifest_path": str(out_tsv),
        "key2txt_tsv": str(key2txt_tsv),
        "key2structure_tsv": str(key2structure_resolved) if key2structure_resolved else "",
        "table_cell_sidecar_manifest_tsv": str(table_cell_sidecar_manifest_resolved) if table_cell_sidecar_manifest_resolved else "",
        "mainline_integration_status": "maintained_code_only_not_hydrated",
        "dataset_sources": [
            {
                "dataset_manifest_tsv": str(dataset_manifests[idx]),
                "dataset_id": dataset_ids[idx],
                "dataset_tables_root": str(dataset_tables_map[dataset_ids[idx]]),
            }
            for idx in range(len(dataset_manifests))
        ],
        "split_sources": [
            {
                "split_manifest_tsv": str(split_manifests[idx]),
                "split_tag": split_tags[idx],
                "benchmark_tag": benchmark_tags[idx],
            }
            for idx in range(len(split_manifests))
        ],
        "row_count": len(rows),
        "text_available_yes_count": 0,
        "table_available_yes_count": 0,
        "structure_available_yes_count": 0,
        "stage1_table_cell_sidecar_available_yes_count": 0,
        "dataset_overlay_count": 0,
        "split_overlay_count": 0,
        "benchmark_overlay_count": 0,
        "dataset_conflicts": dataset_conflicts,
        "split_conflicts": split_conflicts,
    }

    for row in rows:
        hydrated = dict(row)
        key = normalize_key(row.get("key") or row.get("paper_key") or row.get("zotero_key"))

        text_path = key2txt_map.get(key, "")
        hydrated["text_path"] = text_path
        hydrated["text_source_type"] = infer_text_source_type(text_path)
        hydrated["text_available"] = "yes" if text_path else "no"
        if hydrated["text_available"] == "yes":
            hydration_summary["text_available_yes_count"] += 1

        structure_path = key2structure_map.get(key, "")
        hydrated["structure_path"] = structure_path
        hydrated["structure_available"] = "yes" if structure_path else "no"
        if hydrated["structure_available"] == "yes":
            hydration_summary["structure_available_yes_count"] += 1

        sidecar_path, sidecar_available = table_cell_sidecar_map.get(key, ("", "no"))
        hydrated["stage1_table_cell_sidecar_path"] = sidecar_path
        hydrated["stage1_table_cell_sidecar_available"] = sidecar_available if sidecar_path else "no"
        if hydrated["stage1_table_cell_sidecar_available"].lower() == "yes":
            hydration_summary["stage1_table_cell_sidecar_available_yes_count"] += 1

        if dataset_manifests:
            dataset_id = dataset_membership.get(key, "")
            hydrated["dataset_id"] = dataset_id
            if dataset_id:
                hydration_summary["dataset_overlay_count"] += 1
                table_dir_path = dataset_tables_map[dataset_id] / key
                if table_dir_path.exists():
                    hydrated["table_dir"] = to_repo_rel_windows(table_dir_path)
                    hydrated["table_available"] = "yes"
                    hydration_summary["table_available_yes_count"] += 1
                else:
                    hydrated["table_dir"] = ""
                    hydrated["table_available"] = "no"
            else:
                hydrated["table_dir"] = ""
                hydrated["table_available"] = "no"
        else:
            dataset_id = normalize_text(hydrated.get("dataset_id"))
            hydrated["dataset_id"] = dataset_id
            hydrated["table_dir"] = normalize_text(hydrated.get("table_dir"))
            hydrated["table_available"] = normalize_text(hydrated.get("table_available")) or ("yes" if hydrated["table_dir"] else "no")
            if dataset_id:
                hydration_summary["dataset_overlay_count"] += 1
            if hydrated["table_available"].lower() == "yes":
                hydration_summary["table_available_yes_count"] += 1

        if split_manifests:
            split_tag, benchmark_tag = split_overlay.get(key, ("", ""))
            hydrated["split_tag"] = split_tag
            hydrated["benchmark_tag"] = benchmark_tag
        else:
            split_tag = normalize_text(hydrated.get("split_tag"))
            benchmark_tag = normalize_text(hydrated.get("benchmark_tag"))
            hydrated["split_tag"] = split_tag
            hydrated["benchmark_tag"] = benchmark_tag
        if split_tag:
            hydration_summary["split_overlay_count"] += 1
        if benchmark_tag:
            hydration_summary["benchmark_overlay_count"] += 1

        hydrated_rows.append(hydrated)

    out_fieldnames = list(fieldnames)
    for required_field in (
        "text_path",
        "text_source_type",
        "text_available",
        "table_dir",
        "table_available",
        "structure_path",
        "structure_available",
        "stage1_table_cell_sidecar_path",
        "stage1_table_cell_sidecar_available",
        "dataset_id",
        "split_tag",
        "benchmark_tag",
    ):
        if required_field not in out_fieldnames:
            out_fieldnames.append(required_field)

    write_tsv(out_tsv, out_fieldnames, hydrated_rows)

    metadata_path = metadata_json
    if metadata_path is None:
        metadata_path = out_tsv.with_name(f"{out_tsv.stem}__hydration_metadata_v1.json")
    elif not metadata_path.is_absolute():
        metadata_path = (paths.PROJECT_ROOT / metadata_path).resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(hydration_summary, indent=2), encoding="utf-8")
    return hydration_summary


def main() -> None:
    args = parse_args()
    hydration_summary = hydrate_manifest(
        manifest_tsv=args.manifest_tsv,
        out_tsv=args.out_tsv,
        key2txt_tsv=args.key2txt_tsv,
        key2structure_tsv=args.key2structure_tsv,
        table_cell_sidecar_manifest_tsv=args.table_cell_sidecar_manifest_tsv,
        dataset_manifests=args.dataset_manifests,
        dataset_ids=args.dataset_ids,
        dataset_tables_roots=args.dataset_tables_roots,
        split_manifests=args.split_manifests,
        split_tags=args.split_tags,
        benchmark_tags=args.benchmark_tags,
        metadata_json=args.metadata_json,
        strict_overlays=args.strict_overlays,
        overwrite=args.overwrite,
    )

    print(f"input_manifest={hydration_summary['input_manifest_path']}")
    print(f"hydrated_manifest={hydration_summary['output_manifest_path']}")
    print(f"row_count={hydration_summary['row_count']}")
    print(f"text_available_yes_count={hydration_summary['text_available_yes_count']}")
    print(f"table_available_yes_count={hydration_summary['table_available_yes_count']}")
    print(f"structure_available_yes_count={hydration_summary['structure_available_yes_count']}")
    print(
        "stage1_table_cell_sidecar_available_yes_count="
        f"{hydration_summary['stage1_table_cell_sidecar_available_yes_count']}"
    )

if __name__ == "__main__":
    main()
