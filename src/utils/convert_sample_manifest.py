"""
convert_sample_manifest.py

Purpose:
Convert a sample definition JSONL into the schema expected by
build_key2txt_from_sample_manifest.py, i.e. each record has:
- zotero_key
- cleaned_text_sample

Usage (PowerShell):
  python .\scripts\convert_sample_manifest.py --in-jsonl data\cleaned\samples\sample10_htmlfirst.jsonl --out-jsonl data\cleaned\samples\sample10_htmlfirst_for_key2txt.jsonl

Notes:
- The script tries common field names for the key and cleaned text path.
- If it cannot find suitable fields, it will print a clear error and show the keys it sees.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

KEY_CANDIDATES = ["zotero_key", "key", "paper_key", "item_key", "citation_key"]
PATH_CANDIDATES = [
    "cleaned_text_sample",
    "cleaned_text",
    "txt_path",
    "text_path",
    "cleaned_txt",
    "cleaned_path",
]

def pick_field(d: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in d and d[c] not in (None, ""):
            return c
    return None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True, help="Input sample JSONL file")
    ap.add_argument("--out-jsonl", required=True, help="Output JSONL with required keys")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)

    if not in_path.exists():
        raise SystemExit(f"[ERROR] Input not found: {in_path}")

    lines = in_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit("[ERROR] Input JSONL is empty.")

    converted = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"[ERROR] JSON parse error at line {i}: {e}")

            if not isinstance(obj, dict):
                skipped += 1
                continue

            kf = pick_field(obj, KEY_CANDIDATES)
            pf = pick_field(obj, PATH_CANDIDATES)

            if kf is None or pf is None:
                # show a helpful message once, then continue
                if i == 1:
                    seen = sorted(list(obj.keys()))
                    raise SystemExit(
                        "[ERROR] Cannot find required fields in sample JSONL.\n"
                        f"  Need a key field (candidates: {KEY_CANDIDATES}) and a path field (candidates: {PATH_CANDIDATES}).\n"
                        f"  First record keys seen: {seen}\n"
                        "  Fix by regenerating the sample JSONL with the expected schema, or update this converter's candidate lists."
                    )
                skipped += 1
                continue

            out_obj = {
                "zotero_key": obj[kf],
                "cleaned_text_sample": obj[pf],
            }
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            converted += 1

    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] Records converted: {converted}")
    if skipped:
        print(f"[WARN] Records skipped: {skipped}")

if __name__ == "__main__":
    main()
