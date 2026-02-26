from __future__ import annotations

import argparse
from pathlib import Path


REGISTRY_PATH = Path("project/8_EVAL_SPLITS_REGISTRY.md")


def _extract_tsv_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    lines = text.splitlines()
    in_tsv = False
    cur: list[str] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("```tsv"):
            in_tsv = True
            cur = []
            continue
        if in_tsv and s.startswith("```"):
            in_tsv = False
            if cur:
                blocks.append("\n".join(cur))
            cur = []
            continue
        if in_tsv:
            cur.append(ln)
    return blocks


def _parse_tsv_block(block: str) -> tuple[list[str], list[dict[str, str]]]:
    rows = [ln for ln in block.splitlines() if ln.strip()]
    if not rows:
        return [], []
    headers = [h.strip() for h in rows[0].split("\t")]
    out_rows: list[dict[str, str]] = []
    for ln in rows[1:]:
        parts = [p.strip() for p in ln.split("\t")]
        rec: dict[str, str] = {}
        for i, h in enumerate(headers):
            rec[h] = parts[i] if i < len(parts) else ""
        out_rows.append(rec)
    return headers, out_rows


def _load_registry_text(registry_path: Path = REGISTRY_PATH) -> str:
    if not registry_path.exists():
        raise FileNotFoundError(f"split registry not found: {registry_path}")
    return registry_path.read_text(encoding="utf-8", errors="replace")


def load_registered_dev_keys(dataset_id: str, registry_path: Path = REGISTRY_PATH) -> set[str]:
    text = _load_registry_text(registry_path=registry_path)
    keys: set[str] = set()
    for b in _extract_tsv_blocks(text):
        headers, rows = _parse_tsv_block(b)
        if {"dataset_id", "zotero_key"}.issubset(set(headers)):
            for r in rows:
                if (r.get("dataset_id", "").strip() == dataset_id) and r.get("zotero_key", "").strip():
                    keys.add(r["zotero_key"].strip())
    if not keys:
        raise ValueError(f"no registered DEV keys found for dataset_id={dataset_id} in {registry_path}")
    return keys


def load_registered_dev_keys_file(dataset_id: str, registry_path: Path = REGISTRY_PATH) -> str:
    text = _load_registry_text(registry_path=registry_path)
    lines = text.splitlines()
    start = None
    end = None
    for i, ln in enumerate(lines):
        if ln.strip() == f"## {dataset_id}":
            start = i
            continue
        if start is not None and ln.startswith("## "):
            end = i
            break
    if start is None:
        raise ValueError(f"dataset section not found in registry: {dataset_id}")
    section = lines[start : (end if end is not None else len(lines))]
    for ln in section:
        s = ln.strip()
        if s.startswith("- dev_keys_file:"):
            value = s.split(":", 1)[1].strip().strip("`")
            if value:
                return value
    raise ValueError(f"no dev_keys_file entry found for dataset_id={dataset_id}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Print registered DEV keys from split registry.")
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--registry-path", default=str(REGISTRY_PATH))
    args = ap.parse_args()
    keys = sorted(load_registered_dev_keys(args.dataset_id, registry_path=Path(args.registry_path)))
    for k in keys:
        print(k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
