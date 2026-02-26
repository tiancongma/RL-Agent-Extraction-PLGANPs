from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_MIN_COLUMNS = ("zotero_key", "title", "doi", "year", "pdf", "html", "notes")


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"TSV has no header: {path}")
        rows = [{k: (v or "") for k, v in row.items()} for row in reader]
        return list(reader.fieldnames), rows


def _normalize_key_column(headers: list[str], rows: list[dict[str, str]]) -> tuple[list[str], list[dict[str, str]]]:
    if "zotero_key" in headers:
        return headers, rows
    if "key" in headers:
        new_headers = ["zotero_key" if h == "key" else h for h in headers]
        out_rows: list[dict[str, str]] = []
        for r in rows:
            out = {}
            for h in headers:
                if h == "key":
                    out["zotero_key"] = r.get("key", "")
                else:
                    out[h] = r.get(h, "")
            out_rows.append(out)
        return new_headers, out_rows
    raise ValueError("Input TSV must contain 'zotero_key' or 'key'.")


def _merge_rows(
    headers: list[str], rows: list[dict[str, str]]
) -> tuple[list[str], list[dict[str, str]]]:
    by_key: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get("zotero_key", "").strip()
        if not key:
            continue
        cur = by_key.setdefault(key, {})
        for h in headers:
            v = (row.get(h) or "").strip()
            if v and not (cur.get(h) or "").strip():
                cur[h] = v
            elif h not in cur:
                cur[h] = (row.get(h) or "")
    out_rows = [by_key[k] for k in sorted(by_key.keys())]
    return headers, out_rows


def _ensure_required_columns(headers: list[str]) -> list[str]:
    out = list(headers)
    for c in REQUIRED_MIN_COLUMNS:
        if c not in out:
            out.append(c)
    return out


def _write_tsv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build canonical global Zotero-wide index: data/cleaned/index/manifest__zotero_all.tsv"
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=["data/cleaned/index/manifest_current.tsv"],
        help="Input TSV files (zotero-wide preferred first).",
    )
    ap.add_argument(
        "--out",
        default="data/cleaned/index/manifest__zotero_all.tsv",
        help="Canonical global index output path.",
    )
    args = ap.parse_args()

    all_headers: list[str] = []
    all_rows: list[dict[str, str]] = []
    for s in args.inputs:
        p = Path(s)
        if not p.exists():
            raise FileNotFoundError(f"input not found: {p}")
        headers, rows = _read_tsv(p)
        headers, rows = _normalize_key_column(headers, rows)
        for h in headers:
            if h not in all_headers:
                all_headers.append(h)
        all_rows.extend(rows)

    merged_headers, merged_rows = _merge_rows(all_headers, all_rows)
    merged_headers = _ensure_required_columns(merged_headers)

    out_path = Path(args.out)
    _write_tsv(out_path, merged_headers, merged_rows)

    print("build_global_zotero_index_v1")
    print(f"inputs\t{';'.join(args.inputs)}")
    print(f"out\t{out_path}")
    print(f"n_rows\t{len(merged_rows)}")
    print(f"columns\t{','.join(merged_headers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
