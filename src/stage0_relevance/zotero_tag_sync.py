#!/usr/bin/env python3
"""
zotero_tag_sync.py (robust + verbose + only-key + force-replace)

Adds:
  - Robust column matching (tolerates case/space/BOM): Key/DOI/AI_Tag/llm_*.
  - --verbose: prints per-row diagnosis if no tags were built or updates skipped.
  - --limit N: process only the first N rows.
  - CSV read with encoding='utf-8-sig' to handle BOM.
  - Environment parsing strips inline comments and whitespace.
  - --only-key: process only one Zotero item key.
  - --force-replace: remove existing LLM:* tags before adding new ones.
"""

import argparse
import ast
import os
import sys
import time
from pathlib import Path
from typing import List, Set, Dict

from dotenv import load_dotenv
import pandas as pd
from pyzotero import zotero
from requests.exceptions import HTTPError

# ---------- Utilities ----------

def norm_name(s: str) -> str:
    """Normalize column names for robust matching: lower, strip, replace spaces and hyphens with underscores."""
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in s.strip().lower()).replace("__", "_")

def find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """Find a column by a list of candidate names; returns the real column name or '' if not found."""
    norm_map: Dict[str, str] = {norm_name(c): c for c in df.columns}
    for cand in candidates:
        nc = norm_name(cand)
        if nc in norm_map:
            return norm_map[nc]
    # try fuzzy: any column whose normalized name endswith the candidate (for 'ai_tag ' etc.)
    for c in df.columns:
        if norm_name(c).endswith(norm_name(candidates[0])):
            return c
    return ""

def parse_listish(cell) -> List[str]:
    """Parse list-like strings or delimiter-separated values into a list of strings."""
    if cell is None:
        return []
    if isinstance(cell, float) and pd.isna(cell):
        return []
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return []
    # Try Python literal list
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
    except Exception:
        pass
    # Fallback: split by common delimiters
    parts = []
    for delim in [';', '|', ',']:
        if delim in s:
            parts = [x.strip() for x in s.split(delim)]
            break
    if not parts:
        parts = [s]
    return [x for x in parts if x]

def normalize_str(x) -> str:
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()

def map_relevance(val: str) -> str:
    v = (val or "").strip().lower()
    if not v:
        return ""
    m = {
        "y": "LLM:Relevant",
        "yes": "LLM:Relevant",
        "relevant": "LLM:Relevant",
        "borderline": "LLM:Borderline",
        "n": "LLM:Irrelevant",
        "no": "LLM:Irrelevant",
        "irrelevant": "LLM:Irrelevant",
    }
    return m.get(v, "")

def unique_merge(existing: List[dict], new_tags: List[str]) -> List[dict]:
    existing_set: Set[str] = {t.get("tag", "") for t in existing}
    for t in new_tags:
        if t and t not in existing_set:
            existing.append({"tag": t})
            existing_set.add(t)
    return existing

def rate_limit_sleep(attempt: int):
    delay = min(2 ** attempt, 30)
    time.sleep(delay)

# ---------- Main ----------

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Sync tags from CSV to Zotero items by Key (with optional DOI fallback).")
    parser.add_argument("--csv", required=True, help="Path to the CSV exported from Zotero (augmented with your tag columns).")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done; do not modify Zotero.")
    parser.add_argument("--use-doi-fallback", action="store_true", help="If item Key not found, try exact DOI search as fallback.")
    parser.add_argument("--verbose", action="store_true", help="Print per-row diagnostics for debugging.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N rows (for testing).")
    parser.add_argument("--only-key", help="Only update a single Zotero item key (for testing).")
    parser.add_argument("--force-replace", action="store_true", help="Force replace existing LLM:* tags.")
    args = parser.parse_args()

    # Read env with sanitization
    library_type = (os.getenv("ZOTERO_LIBRARY_TYPE", "user") or "user")
    library_type = library_type.split("#", 1)[0].strip().lower()
    library_id = (os.getenv("ZOTERO_LIBRARY_ID") or "").split("#", 1)[0].strip()
    api_key    = (os.getenv("ZOTERO_API_KEY") or "").split("#", 1)[0].strip()

    if library_type not in {"user", "group"}:
        print(f"ERROR: ZOTERO_LIBRARY_TYPE must be 'user' or 'group', got '{library_type}'", file=sys.stderr)
        sys.exit(1)
    if not library_id or not api_key:
        print("ERROR: Missing ZOTERO_LIBRARY_ID and/or ZOTERO_API_KEY.", file=sys.stderr)
        sys.exit(1)

    # Init Zotero
    try:
        zot = zotero.Zotero(library_id, library_type, api_key)
    except Exception as e:
        print(f"ERROR: Failed to initialize Zotero client: {e}", file=sys.stderr)
        sys.exit(1)

    # Resolve CSV path
    csv_arg = str(args.csv)
    candidates = [Path(csv_arg), Path.cwd() / csv_arg]
    p = Path(csv_arg)
    if p.parent == Path('.'):
        candidates.append(Path(__file__).resolve().parent.parent / 'data' / p.name)

    csv_path = None
    for c in candidates:
        if c.exists():
            csv_path = c
            break
    if csv_path is None:
        tried = "\n  - ".join(str(c) for c in candidates)
        print(
            "ERROR: CSV file not found. Tried:\n"
            f"  - {tried}\n"
            f"(working dir = {Path.cwd()})",
            file=sys.stderr
        )
        sys.exit(1)

    # Load CSV (handle BOM)
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        print(f"ERROR: Failed to read CSV '{csv_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Normalize columns
    col_key = find_col(df, ["Key"])
    col_doi = find_col(df, ["DOI"])
    col_llm_rel = find_col(df, ["llm_is_relevant", "llm relevant", "llm_relevant"])
    col_ai_tag  = find_col(df, ["AI_Tag", "AI Tag", "ai_tag"])
    col_route   = find_col(df, ["llm_route", "route"])
    col_params  = find_col(df, ["llm_has_params", "params", "llm_params"])
    col_results = find_col(df, ["llm_has_results", "results", "llm_results"])

    if not col_key:
        print(f"ERROR: Could not find 'Key' column in CSV. Found columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    total = len(df) if args.limit <= 0 else min(args.limit, len(df))
    print(f"Loaded {len(df)} rows from {csv_path}")
    if args.limit > 0:
        print(f"[INFO] Limiting to first {total} rows as requested by --limit {args.limit}")
    if args.dry_run:
        print("[DRY RUN] No changes will be sent to Zotero.")

    updated = 0
    skipped = 0
    not_found = 0

    for idx in range(total):
        row = df.iloc[idx]
        key = normalize_str(row.get(col_key, ""))

        # --only-key: skip all others
        if args.only_key and key != args.only_key:
            continue

        # Build tags
        tags: List[str] = []

        # Relevance from llm or AI_Tag
        rel_from = ""
        if col_llm_rel:
            rel_from = "llm_is_relevant"
            rel_tag = map_relevance(normalize_str(row.get(col_llm_rel, "")))
            if rel_tag:
                tags.append(rel_tag)
        if not tags and col_ai_tag:
            rel_from = "AI_Tag"
            rel_tag = map_relevance(normalize_str(row.get(col_ai_tag, "")))
            if rel_tag:
                tags.append(rel_tag)

        # Route
        if col_route:
            route = normalize_str(row.get(col_route, ""))
            if route and route.lower() not in {"", "other/irrelevant", "irrelevant", "other"}:
                tags.append(f"Route:{route}")

        # Params
        if col_params:
            for pval in parse_listish(row.get(col_params)):
                tags.append(f"Params:{pval}")

        # Results
        if col_results:
            for rval in parse_listish(row.get(col_results)):
                tags.append(f"Results:{rval}")

        if not tags:
            skipped += 1
            if args.verbose:
                print(f"[{idx}] key={key or '(missing)'} -> SKIP (no tags). "
                      f"Cols: rel({col_llm_rel or 'N/A'}) ai({col_ai_tag or 'N/A'}) route({col_route or 'N/A'}) "
                      f"params({col_params or 'N/A'}) results({col_results or 'N/A'})")
            continue

        # Fetch item
        item = None
        if key:
            attempt = 0
            while True:
                try:
                    item = zot.item(key)
                    break
                except HTTPError as e:
                    if e.response is not None and e.response.status_code in (429, 503):
                        attempt += 1
                        if args.verbose:
                            print(f"[{idx}] Rate-limited fetching key={key}. Backoff attempt {attempt}...")
                        rate_limit_sleep(attempt)
                        continue
                    else:
                        if args.verbose:
                            print(f"[{idx}] HTTP error fetching key={key}: {e}")
                        item = None
                        break
                except Exception as e:
                    if args.verbose:
                        print(f"[{idx}] Error fetching key={key}: {e}")
                    item = None
                    break
        else:
            if args.use_doi_fallback and col_doi:
                doi = normalize_str(row.get(col_doi, "")).lower()
                if doi:
                    try:
                        items = zot.items(q=doi, qmode="exact")
                        if items:
                            item = items[0]
                    except Exception as e:
                        if args.verbose:
                            print(f"[{idx}] DOI lookup failed for {doi}: {e}")

        if item is None:
            not_found += 1
            if args.verbose:
                print(f"[{idx}] key={key or '(missing)'} -> NOT FOUND in library.")
            continue

        # Merge & update
        existing_tags = item["data"].get("tags", [])

        # --force-replace: drop all existing LLM:* before merging
        if args.force_replace:
            before = len(existing_tags)
            existing_tags = [t for t in existing_tags if not t.get("tag", "").startswith("LLM:")]
            removed = before - len(existing_tags)
            if args.verbose and removed:
                print(f"[{idx}] key={item['key']} -> removed {removed} old LLM:* tags due to --force-replace.")

        merged = unique_merge(existing_tags, tags)

        # If not force-replace, skip when LLM:* already present
        if not args.force_replace:
            existing_llm_tags = {t.get("tag", "") for t in existing_tags if t.get("tag", "").startswith("LLM:")}
            new_llm_tags = {t for t in tags if t.startswith("LLM:")}
            need_update = not new_llm_tags.issubset(existing_llm_tags)
            if not need_update:
                skipped += 1
                if args.verbose:
                    print(f"[{idx}] key={item['key']} -> SKIP (LLM tags already present). Existing={existing_llm_tags}, Built={tags}")
                continue
        else:
            # With force-replace, still skip if nothing actually changes
            if merged == item["data"].get("tags", []):
                skipped += 1
                if args.verbose:
                    print(f"[{idx}] key={item['key']} -> SKIP (no changes after force-replace). Built={tags}")
                continue

        if args.dry_run:
            updated += 1
            print(f"[DRY RUN] Would add tags to key={item['key']}: {tags}")
            continue

        attempt = 0
        while True:
            try:
                item["data"]["tags"] = merged
                zot.update_item(item)
                updated += 1
                if args.verbose:
                    print(f"[{idx}] key={item['key']} -> UPDATED: +{tags}")
                break
            except HTTPError as e:
                if e.response is not None and e.response.status_code in (429, 503):
                    attempt += 1
                    if args.verbose:
                        print(f"[{idx}] Rate-limited updating key={item['key']}. Backoff attempt {attempt}...")
                    rate_limit_sleep(attempt)
                    continue
                else:
                    if args.verbose:
                        print(f"[{idx}] HTTP error updating key={item['key']}: {e}")
                    skipped += 1
                    break
            except Exception as e:
                if args.verbose:
                    print(f"[{idx}] Error updating key={item['key']}: {e}")
                skipped += 1
                break

    print(f"Done. Updated={updated}, Skipped={skipped}, NotFound={not_found}, Total={total}")

if __name__ == "__main__":
    main()
