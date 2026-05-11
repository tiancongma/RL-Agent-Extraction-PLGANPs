from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

try:
    from src.utils.paths import DATA_CLEANED_DIR, dataset_cleaned_root
    from src.utils.split_registry_v1 import (
        REGISTRY_PATH,
        load_registered_dev_keys,
        load_registered_dev_keys_file,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import DATA_CLEANED_DIR, dataset_cleaned_root
    from src.utils.split_registry_v1 import (
        REGISTRY_PATH,
        load_registered_dev_keys,
        load_registered_dev_keys_file,
    )


ALLOWED_TOPLEVEL = {"index", "content", "text", "sections", "tables", "analysis"}
ZOTERO_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")


def _iter_dirs(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: p.name)


def _validate_per_key_dirs(asset_root: Path, asset_name: str, rows: list[dict], issues: list[str]) -> None:
    for d in _iter_dirs(asset_root):
        ok = bool(ZOTERO_KEY_RE.fullmatch(d.name))
        rows.append(
            {
                "check": f"{asset_name}_key_dir",
                "path": str(d),
                "status": "OK" if ok else "WARN",
                "detail": "" if ok else f"non_zotero_key_dir_name={d.name}",
            }
        )
        if not ok:
            issues.append(f"{asset_name}: invalid key directory name: {d.name}")


def validate_layout(
    dataset_root: Path,
    check_keys: bool,
    strict_global: bool = False,
    check_coverage_reasons: bool = False,
    coverage_tsv: Path | None = None,
    check_split_registry: bool = False,
    dataset_id: str = "",
    registry_path: Path = REGISTRY_PATH,
) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    issues: list[str] = []
    global_index = DATA_CLEANED_DIR / "index" / "manifest__zotero_all.tsv"
    global_ok = global_index.exists() and global_index.is_file()
    rows.append(
        {
            "check": "global_index_exists",
            "path": str(global_index),
            "status": "OK" if global_ok else "WARN",
            "detail": "",
        }
    )
    if strict_global and not global_ok:
        issues.append(f"missing global index: {global_index}")

    rows.append(
        {
            "check": "dataset_root_exists",
            "path": str(dataset_root),
            "status": "OK" if dataset_root.exists() and dataset_root.is_dir() else "ERROR",
            "detail": "",
        }
    )
    if not dataset_root.exists() or not dataset_root.is_dir():
        issues.append(f"dataset root missing: {dataset_root}")
        return rows, issues

    for name in ("index", "text", "sections", "tables"):
        p = dataset_root / name
        ok = p.exists() and p.is_dir()
        rows.append(
            {
                "check": "required_subdir",
                "path": str(p),
                "status": "OK" if ok else "ERROR",
                "detail": f"name={name}",
            }
        )
        if not ok:
            issues.append(f"missing required subdir: {name}")
    dataset_manifest = dataset_root / "index" / "manifest.tsv"
    manifest_ok = dataset_manifest.exists() and dataset_manifest.is_file()
    rows.append(
        {
            "check": "dataset_manifest_exists",
            "path": str(dataset_manifest),
            "status": "OK" if manifest_ok else "ERROR",
            "detail": "",
        }
    )
    if not manifest_ok:
        issues.append(f"missing dataset manifest: {dataset_manifest}")

    unexpected = [
        p for p in _iter_dirs(dataset_root) if p.name not in ALLOWED_TOPLEVEL
    ]
    for p in unexpected:
        rows.append(
            {
                "check": "unexpected_top_level_dir",
                "path": str(p),
                "status": "WARN",
                "detail": "not_in_allowlist",
            }
        )
        issues.append(f"unexpected top-level dir: {p.name}")

    if check_keys:
        tables_root = dataset_root / "tables"
        text_root = dataset_root / "text"
        sections_root = dataset_root / "sections"

        if tables_root.exists() and tables_root.is_dir():
            # Bug-prevention check: tables must be paper-local under tables/<key>/.
            csv_direct = sorted(tables_root.glob("*.csv"))
            rows.append(
                {
                    "check": "tables_paper_local_csv",
                    "path": str(tables_root),
                    "status": "OK" if not csv_direct else "WARN",
                    "detail": "no_direct_csv_under_tables_root" if not csv_direct else f"direct_csv_count={len(csv_direct)}",
                }
            )
            if csv_direct:
                issues.append(
                    f"tables root has non-paper-local csv files: {len(csv_direct)}"
                )
            _validate_per_key_dirs(tables_root, "tables", rows, issues)

        if text_root.exists() and text_root.is_dir():
            _validate_per_key_dirs(text_root, "text", rows, issues)

        if sections_root.exists() and sections_root.is_dir():
            _validate_per_key_dirs(sections_root, "sections", rows, issues)

    if check_coverage_reasons:
        cov = coverage_tsv if coverage_tsv is not None else dataset_root / "analysis" / "tables_extraction_coverage.tsv"
        cov_exists = cov.exists() and cov.is_file()
        rows.append(
            {
                "check": "coverage_tsv_exists",
                "path": str(cov),
                "status": "OK" if cov_exists else "WARN",
                "detail": "",
            }
        )
        if cov_exists:
            bad = 0
            with cov.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for rec in reader:
                    pdf_found = str(rec.get("pdf_found", "")).strip().lower() in {"true", "1", "yes"}
                    n_pdf = int(str(rec.get("n_tables_pdf_extracted", "0") or "0"))
                    pdf_reason = str(rec.get("pdf_table_reason", "") or "").strip()
                    if pdf_found and n_pdf == 0 and not pdf_reason:
                        bad += 1
            rows.append(
                {
                    "check": "coverage_pdf_reason_constraint",
                    "path": str(cov),
                    "status": "OK" if bad == 0 else "WARN",
                    "detail": f"violations={bad}",
                }
            )
            if bad > 0:
                issues.append(f"coverage pdf_table_reason violations: {bad}")

    if check_split_registry:
        reg_exists = registry_path.exists() and registry_path.is_file()
        rows.append(
            {
                "check": "split_registry_exists",
                "path": str(registry_path),
                "status": "OK" if reg_exists else "WARN",
                "detail": "",
            }
        )
        if not reg_exists:
            issues.append(f"split registry missing: {registry_path}")
        else:
            try:
                registry_keys = load_registered_dev_keys(dataset_id, registry_path=registry_path)
                dev_keys_file = load_registered_dev_keys_file(dataset_id, registry_path=registry_path)
                dev_path = Path(dev_keys_file)
                rows.append(
                    {
                        "check": "split_registry_dev_keys_file_exists",
                        "path": str(dev_path),
                        "status": "OK" if dev_path.exists() and dev_path.is_file() else "WARN",
                        "detail": "",
                    }
                )
                if not (dev_path.exists() and dev_path.is_file()):
                    issues.append(f"dev keys file missing from registry path: {dev_path}")
                else:
                    with dev_path.open("r", encoding="utf-8", newline="") as f:
                        reader = csv.DictReader(f, delimiter="\t")
                        file_keys = {
                            str(r.get("zotero_key", "")).strip()
                            for r in reader
                            if str(r.get("zotero_key", "")).strip()
                        }
                    matched = registry_keys == file_keys
                    rows.append(
                        {
                            "check": "split_registry_key_match",
                            "path": str(dev_path),
                            "status": "OK" if matched else "WARN",
                            "detail": f"registry_n={len(registry_keys)} file_n={len(file_keys)}",
                        }
                    )
                    if not matched:
                        issues.append(
                            f"split registry keys mismatch: registry_n={len(registry_keys)} file_n={len(file_keys)}"
                        )
            except Exception as e:
                issues.append(f"split registry parse error: {type(e).__name__}: {e}")
                rows.append(
                    {
                        "check": "split_registry_parse",
                        "path": str(registry_path),
                        "status": "WARN",
                        "detail": f"{type(e).__name__}:{e}",
                    }
                )

    return rows, issues


def _write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["check", "path", "status", "detail"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate cleaned dataset layout against convention v1."
    )
    ap.add_argument("--dataset-id", required=True, help="Dataset id under data/cleaned/")
    ap.add_argument(
        "--cleaned-root",
        default=str(DATA_CLEANED_DIR),
        help="Cleaned root directory (default: data/cleaned)",
    )
    ap.add_argument(
        "--check-keys",
        action="store_true",
        help="Validate per-key directory naming and paper-local tables shape.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any ERROR/WARN issues are found.",
    )
    ap.add_argument(
        "--strict-global",
        action="store_true",
        help="Treat missing global index data/cleaned/index/manifest__zotero_all.tsv as an error.",
    )
    ap.add_argument(
        "--write-tsv",
        action="store_true",
        help="Write report TSV to data/cleaned/<dataset_id>/analysis/dataset_layout_validation.tsv",
    )
    ap.add_argument(
        "--out-tsv",
        default="",
        help="Optional explicit TSV report path.",
    )
    ap.add_argument(
        "--check-coverage-reasons",
        action="store_true",
        help="Check coverage constraint: pdf_found=True and n_tables_pdf_extracted=0 requires non-empty pdf_table_reason.",
    )
    ap.add_argument(
        "--coverage-tsv",
        default="",
        help="Optional explicit coverage TSV path (default: data/cleaned/<dataset_id>/analysis/tables_extraction_coverage.tsv).",
    )
    ap.add_argument(
        "--check-split-registry",
        action="store_true",
        help="Verify split registry and dev key file consistency for dataset_id.",
    )
    ap.add_argument(
        "--registry-path",
        default=str(REGISTRY_PATH),
        help="Path to split registry markdown file.",
    )
    args = ap.parse_args()

    cleaned_root = Path(args.cleaned_root)
    dataset_root = (
        dataset_cleaned_root(args.dataset_id)
        if cleaned_root.resolve() == DATA_CLEANED_DIR.resolve()
        else cleaned_root / args.dataset_id
    )

    rows, issues = validate_layout(
        dataset_root,
        check_keys=args.check_keys,
        strict_global=args.strict_global,
        check_coverage_reasons=args.check_coverage_reasons,
        coverage_tsv=Path(args.coverage_tsv) if args.coverage_tsv else None,
        check_split_registry=args.check_split_registry,
        dataset_id=args.dataset_id,
        registry_path=Path(args.registry_path),
    )

    print("dataset_layout_validation_v1")
    print(f"dataset_root\t{dataset_root}")
    print(f"n_checks\t{len(rows)}")
    print(f"n_issues\t{len(issues)}")
    if issues:
        print("issues:")
        for i in issues:
            print(f"- {i}")

    out_tsv = Path(args.out_tsv) if args.out_tsv else None
    if args.write_tsv and out_tsv is None:
        out_tsv = dataset_root / "analysis" / "dataset_layout_validation.tsv"
    if out_tsv:
        _write_tsv(out_tsv, rows)
        print(f"report_tsv\t{out_tsv}")

    if args.strict and issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
