from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from random import Random


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "src" / "utils" / "split_registry_v1.py").exists():
            sys.path.insert(0, str(p))
            return


_bootstrap_import_paths()
from src.utils.split_registry_v1 import load_registered_dev_keys


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise ValueError(f"TSV has no header: {path}")
        rows = [{k: (v or "") for k, v in row.items()} for row in r]
        return list(r.fieldnames), rows


def _write_tsv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def _parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y"}


def _load_dev_keys_file(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"dev keys file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return {str(row.get("zotero_key", "")).strip() for row in r if str(row.get("zotero_key", "")).strip()}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build TEST split excluding registered DEV keys.")
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", default="")
    ap.add_argument(
        "--enforce-split-registry",
        default="true",
        help="true/false; when true, require DEV exclusions from registry or --dev-keys-file.",
    )
    ap.add_argument(
        "--exclude-keys-file",
        default="",
        help="Preferred exclusion key list path for TEST builder (one TSV with zotero_key column).",
    )
    ap.add_argument("--dev-keys-file", default="", help="Optional DEV keys TSV path.")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    dataset_id = args.dataset_id.strip()
    if not dataset_id:
        raise ValueError("dataset-id is empty")
    enforce = _parse_bool(args.enforce_split_registry)
    dataset_root = Path("data/cleaned") / dataset_id
    manifest_path = dataset_root / "index" / "manifest.tsv"
    out_dir = Path(args.out_dir) if args.out_dir else dataset_root / "index" / "splits"
    test_keys_path = out_dir / "test_keys_v1.tsv"
    test_manifest_path = out_dir / "test_manifest_v1.tsv"

    headers, rows = _read_tsv(manifest_path)
    key_col = "zotero_key" if "zotero_key" in headers else ("key" if "key" in headers else None)
    if not key_col:
        raise ValueError("manifest must contain zotero_key or key")

    by_key: dict[str, dict[str, str]] = {}
    for r in rows:
        k = (r.get(key_col) or "").strip()
        if k and k not in by_key:
            by_key[k] = r

    dev_keys: set[str] = set()
    if args.exclude_keys_file.strip():
        dev_keys = _load_dev_keys_file(Path(args.exclude_keys_file))
    elif args.dev_keys_file.strip():
        dev_keys = _load_dev_keys_file(Path(args.dev_keys_file))
    elif enforce:
        dev_keys = load_registered_dev_keys(dataset_id)
    elif args.strict:
        raise ValueError("strict mode requires DEV keys via registry or --dev-keys-file")

    if enforce and not dev_keys:
        raise ValueError("enforce-split-registry=true but DEV keys could not be loaded")

    pool = sorted([k for k in by_key.keys() if k not in dev_keys])
    if args.n > len(pool):
        raise ValueError(f"requested n={args.n} exceeds available non-DEV keys={len(pool)}")
    rnd = Random(args.seed)
    shuffled = list(pool)
    rnd.shuffle(shuffled)
    selected = sorted(shuffled[: args.n])

    overlap = len(set(selected).intersection(dev_keys))
    if args.strict and overlap > 0:
        raise ValueError(f"TEST split overlaps DEV keys: overlap={overlap}")

    test_key_rows = [{"zotero_key": k, "seed": str(args.seed)} for k in selected]
    test_manifest_rows = [by_key[k] for k in selected]

    if not args.dry_run:
        _write_tsv(test_keys_path, ["zotero_key", "seed"], test_key_rows)
        _write_tsv(test_manifest_path, headers, test_manifest_rows)

    print(f"dataset_id\t{dataset_id}")
    print(f"n_selected\t{len(selected)}")
    print(f"dev_excluded_n\t{len(dev_keys)}")
    print(f"dev_overlap_count\t{overlap}")
    print("first_20_test_keys")
    for k in selected[:20]:
        print(k)
    print(f"test_keys_path\t{test_keys_path}")
    print(f"test_manifest_path\t{test_manifest_path}")
    print(f"dry_run\t{args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
