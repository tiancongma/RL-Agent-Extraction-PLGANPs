from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from random import Random


def _truthy(v: str) -> bool:
    return str(v or "").strip().lower() in {"true", "1", "yes", "y"}


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build fixed DEV split for dataset from tables coverage.")
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--manifest-tsv", default="")
    ap.add_argument("--coverage-tsv", default="")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    dataset_id = args.dataset_id.strip()
    if not dataset_id:
        raise ValueError("dataset-id is empty")

    dataset_root = Path("data/cleaned") / dataset_id
    manifest_tsv = Path(args.manifest_tsv) if args.manifest_tsv else dataset_root / "index" / "manifest.tsv"
    coverage_tsv = (
        Path(args.coverage_tsv)
        if args.coverage_tsv
        else dataset_root / "analysis" / "tables_extraction_coverage.tsv"
    )
    out_dir = Path(args.out_dir) if args.out_dir else dataset_root / "index" / "splits"
    dev_keys_path = out_dir / "dev_keys_v1.tsv"
    dev_manifest_path = out_dir / "dev_manifest_v1.tsv"
    readme_path = out_dir / "splits_readme.md"

    m_headers, m_rows = _read_tsv(manifest_tsv)
    c_headers, c_rows = _read_tsv(coverage_tsv)
    if "zotero_key" not in c_headers:
        raise ValueError("coverage TSV must contain zotero_key column")

    manifest_by_key: dict[str, dict[str, str]] = {}
    key_col = "zotero_key" if "zotero_key" in m_headers else ("key" if "key" in m_headers else None)
    if not key_col:
        raise ValueError("manifest TSV must contain zotero_key or key column")
    for r in m_rows:
        k = (r.get(key_col) or "").strip()
        if k and k not in manifest_by_key:
            manifest_by_key[k] = r

    cov_by_key: dict[str, dict[str, str]] = {}
    for r in c_rows:
        k = (r.get("zotero_key") or "").strip()
        if k and k not in cov_by_key:
            cov_by_key[k] = r

    html_keys = sorted([k for k, r in cov_by_key.items() if _truthy(r.get("html_found", ""))])
    pdf_only = sorted(
        [
            k
            for k, r in cov_by_key.items()
            if (not _truthy(r.get("html_found", ""))) and _truthy(r.get("pdf_found", ""))
        ]
    )
    if len(html_keys) > args.n:
        raise ValueError(
            f"html_found keys exceed requested n={args.n}; got {len(html_keys)}. Increase --n."
        )

    remaining = args.n - len(html_keys)
    if remaining > len(pdf_only):
        raise ValueError(
            f"Not enough pdf-only keys to fill DEV set: need {remaining}, have {len(pdf_only)}"
        )

    rnd = Random(args.seed)
    # Deterministic: shuffle sorted list with fixed seed, then take head.
    pdf_pool = list(pdf_only)
    rnd.shuffle(pdf_pool)
    selected_pdf = sorted(pdf_pool[:remaining])
    selected = html_keys + selected_pdf
    selected = sorted(dict.fromkeys(selected))
    if len(selected) != args.n:
        raise ValueError(f"selected key count mismatch: expected {args.n}, got {len(selected)}")

    now = datetime.now(timezone.utc).isoformat()
    dev_rows: list[dict[str, str]] = []
    manifest_rows_selected: list[dict[str, str]] = []
    for k in selected:
        m = manifest_by_key.get(k, {})
        if not m:
            raise ValueError(f"key missing from manifest: {k}")
        reason = "html_found" if k in html_keys else "pdf_only_fallback"
        dev_rows.append(
            {
                "zotero_key": k,
                "doi": m.get("doi", ""),
                "title": m.get("title", ""),
                "year": m.get("year", ""),
                "inclusion_reason": reason,
                "seed": str(args.seed),
                "created_at": now,
            }
        )
        manifest_rows_selected.append(dict(m))

    if not args.dry_run:
        _write_tsv(
            dev_keys_path,
            ["zotero_key", "doi", "title", "year", "inclusion_reason", "seed", "created_at"],
            dev_rows,
        )
        _write_tsv(dev_manifest_path, m_headers, manifest_rows_selected)
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        readme_path.write_text(
            "\n".join(
                [
                    "# Dataset Splits (DEV v1)",
                    "",
                    "This DEV split is used for iterative pipeline improvement only.",
                    "Selection rule: include all keys with html_found=True from tables coverage; then add",
                    "pdf_found=True and html_found=False keys until N is reached.",
                    "Policy: all future TEST sets must exclude keys listed in dev_keys_v1.tsv to prevent leakage.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    for k in selected:
        print(k)
    print(f"n_html_included\t{len(html_keys)}")
    print(f"n_pdf_only_included\t{len(selected_pdf)}")
    print(f"dev_keys_path\t{dev_keys_path}")
    print(f"dev_manifest_path\t{dev_manifest_path}")
    print(f"splits_readme_path\t{readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
