#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sample30_from_manifest.py

Randomly sample N cleaned entries from manifest and produce:
  - data/cleaned/samples/sample30.jsonl
  - data/cleaned/samples/key2txt.tsv
Optionally: subset an existing ground truth JSONL to those keys.

Input manifest: data/cleaned/manifests/zotero_llm_relevant.jsonl (by default)
Each manifest line is expected to include:
  - "zotero_key" (or "key")
  - "status" starting with "OK_" or equal to "OK"
  - "paths": { "text": "<path-to-cleaned.txt>" }
"""

import argparse
import json
import random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/cleaned/manifests/zotero_llm_relevant.jsonl",
                    help="Path to the main JSONL manifest.")
    ap.add_argument("--outdir", default="data/cleaned/samples",
                    help="Directory to store outputs.")
    ap.add_argument("--n", type=int, default=30,
                    help="Number of items to sample.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility.")
    ap.add_argument("--ground-truth", default=None,
                    help="Optional JSONL with records like {'key':..., 'fields':{...}} to subset.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pool = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            status = str(rec.get("status", ""))
            if not (status.startswith("OK_") or status == "OK"):
                continue
            text_path = rec.get("paths", {}).get("text")
            key = rec.get("zotero_key") or rec.get("key")
            if key and text_path and Path(text_path).exists():
                pool.append({
                    "key": key,
                    "title": rec.get("title", ""),
                    "year": rec.get("year", ""),
                    "doi": rec.get("doi", ""),
                    "text_path": text_path
                })

    if not pool:
        raise SystemExit("No eligible OK_* entries with existing 'paths.text' found in manifest.")

    random.seed(args.seed)
    random.shuffle(pool)
    sample = pool[:args.n]

    # 1) Write the subset JSONL (mini manifest)
    sample_jsonl = outdir / "sample30.jsonl"
    with sample_jsonl.open("w", encoding="utf-8") as w:
        for r in sample:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2) Write key→txt index
    key2txt = outdir / "key2txt.tsv"
    with key2txt.open("w", encoding="utf-8") as w:
        for r in sample:
            w.write(f"{r['key']}\t{r['text_path']}\n")

    # 3) Optional: subset the provided ground truth to sampled keys
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        if gt_path.exists():
            keep_keys = {r["key"] for r in sample}
            out_gt = outdir / "ground_truth.jsonl"
            kept = 0
            with gt_path.open("r", encoding="utf-8") as fin, out_gt.open("w", encoding="utf-8") as fout:
                for line in fin:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("key") in keep_keys:
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        kept += 1
            print(f"[INFO] Ground truth subset saved: {out_gt} (kept {kept})")
        else:
            print(f"[WARN] --ground-truth not found: {gt_path}")

    print(f"[OK] sample -> {sample_jsonl}")
    print(f"[OK] key2txt -> {key2txt}")
    print(f"[INFO] Sample size: {len(sample)}")

if __name__ == "__main__":
    main()
