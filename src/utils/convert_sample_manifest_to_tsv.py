"""
convert_sample_manifest_to_tsv.py

Purpose:
Convert a sample JSONL file into a TSV with the exact columns expected by
build_key2txt_from_sample_manifest.py:

  zotero_key    cleaned_text_sample

Why:
Some pipelines expect a TSV (tabular) manifest rather than JSONL records.

Usage (PowerShell, from repo root):
  python .\scripts\convert_sample_manifest_to_tsv.py ^
    --in-jsonl data\cleaned\samples\sample10_htmlfirst.jsonl ^
    --out-tsv  data\cleaned\samples\sample10_for_key2txt.tsv

Then run:
  python .\src\stage2_sampling_labels\build_key2txt_from_sample_manifest.py ^
    --sample-manifest data\cleaned\samples\sample10_for_key2txt.tsv ^
    --output data\cleaned\index\key2txt.tsv
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

KEY_CANDIDATES = ["zotero_key", "key", "paper_key", "item_key", "citation_key"]
PATH_CANDIDATES = ["cleaned_text_sample", "text_path", "txt_path", "cleaned_text", "cleaned_txt"]

def pick_field(d: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in d and d[c] not in (None, ""):
            return c
    return None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True, help="Input sample JSONL file")
    ap.add_argument("--out-tsv", required=True, help="Output TSV with required columns")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_tsv)

    if not in_path.exists():
        raise SystemExit(f"[ERROR] Input not found: {in_path}")

    lines = in_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit("[ERROR] Input JSONL is empty.")

    rows = []
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"[ERROR] JSON parse error at line {i}: {e}")

        if not isinstance(obj, dict):
            continue

        kf = pick_field(obj, KEY_CANDIDATES)
        pf = pick_field(obj, PATH_CANDIDATES)

        if kf is None or pf is None:
            if i == 1:
                seen = sorted(list(obj.keys()))
                raise SystemExit(
                    "[ERROR] Cannot find required fields in sample JSONL.\n"
                    f"  Need a key field (candidates: {KEY_CANDIDATES}) and a path field (candidates: {PATH_CANDIDATES}).\n"
                    f"  First record keys seen: {seen}\n"
                )
            continue

        rows.append((str(obj[kf]).strip(), str(obj[pf]).strip()))

    if not rows:
        raise SystemExit("[ERROR] No rows converted. Check input JSONL format.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["zotero_key", "cleaned_text_sample"])
        w.writerows(rows)

    print(f"[OK] Wrote TSV: {out_path}")
    print(f"[OK] Rows: {len(rows)}")

if __name__ == "__main__":
    main()
