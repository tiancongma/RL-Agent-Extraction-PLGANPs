#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive helper for LLM:Relevant items:

- Auto-lists Zotero items with tag="LLM:Relevant" that have NO PDF and (optionally) do NOT have a skip-tag.
- For each item:
    * Opens its DOI landing page (fallback to item URL if no DOI).
    * You click "Download PDF" in the browser.
    * Script watches your Downloads folder; when a new stable PDF appears, it attaches it to the SAME item (no duplicate parent).
- If you choose to skip an item, the script tags it with a skip-tag (default: "needs-manual"), so it won't show next time.

Environment (.env supported):
  ZOTERO_API_KEY       (required)
  ZOTERO_LIBRARY_ID    (required; numeric)
  ZOTERO_LIBRARY_TYPE  (optional; default "user")
"""

import os
import re
import time
import argparse
import webbrowser
from pathlib import Path
from urllib.parse import quote

from pyzotero import zotero

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------- Defaults --------
DEFAULT_QUERY_TAG = "LLM:Relevant"
DEFAULT_SKIP_TAG  = "needs-manual"
DEFAULT_MIN_PDF_SIZE = 80 * 1024   # 80 KB
DEFAULT_TIMEOUT = 300              # seconds to wait for a download per item
DEFAULT_POLL = 0.3                 # seconds
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)

# -------- Utilities --------
def detect_downloads_dir(cli_path: str | None) -> Path:
    if cli_path:
        p = Path(cli_path).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"ERROR: Downloads folder not found: {p}")
        return p
    home = Path.home()
    for cand in [home / "Downloads", home / "Download"]:
        if cand.exists():
            return cand
    raise SystemExit("ERROR: Could not auto-detect Downloads folder. Pass --downloads <path>.")

def has_pdf_attachment(zot, item_key: str) -> bool:
    try:
        for ch in zot.children(item_key):
            d = ch.get("data", {})
            if d.get("itemType") == "attachment" and (d.get("contentType") or "").lower() == "application/pdf":
                return True
    except Exception:
        pass
    return False

def item_has_tag(item: dict, tag_text: str) -> bool:
    if not tag_text:
        return False
    tags = (item.get("data", {}).get("tags") or [])
    return any((t.get("tag") or "").strip().lower() == tag_text.strip().lower() for t in tags)

def add_tag_to_item(zot, item: dict, tag_text: str) -> None:
    """Append a tag to the item (avoid duplicates) and update the item in-place."""
    d = item.get("data", {})
    tags = list(d.get("tags") or [])
    if any((t.get("tag") or "").strip().lower() == tag_text.strip().lower() for t in tags):
        return  # already has the tag

    tags.append({"tag": tag_text})
    # mutate the original item object
    item["data"]["tags"] = tags

    # try update; if version conflict, refetch and retry once
    try:
        zot.update_item(item)
    except Exception:
        try:
            fresh = zot.item(d["key"])
            ftags = list(fresh["data"].get("tags") or [])
            if not any((t.get("tag") or "").strip().lower() == tag_text.strip().lower() for t in ftags):
                ftags.append({"tag": tag_text})
            fresh["data"]["tags"] = ftags
            zot.update_item(fresh)
        except Exception as e:
            # fail to raise e
            raise e


def extract_doi(item: dict) -> str | None:
    data = item.get("data", {})
    doi = (data.get("DOI") or "").strip()
    if doi and DOI_RE.match(doi):
        return doi
    extra = data.get("extra") or ""
    m = re.search(r"\bDOI:\s*(10\.\S+)", extra, re.IGNORECASE)
    if m and DOI_RE.match(m.group(1)):
        return m.group(1)
    return None

def open_item_in_browser(item: dict) -> str | None:
    """Open DOI (preferred) or fallback to item URL; return opened URL or None."""
    doi = extract_doi(item)
    if doi:
        url = f"https://doi.org/{quote(doi)}"
        webbrowser.open_new_tab(url)
        return url
    url = (item.get("data", {}).get("url") or "").strip()
    if url:
        webbrowser.open_new_tab(url)
        return url
    return None

def newest_pdf_since(download_dir: Path, since_ts: float, min_size: int) -> Path | None:
    newest = None
    newest_mtime = None
    for p in download_dir.glob("*.pdf"):
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        if st.st_mtime >= since_ts and st.st_size >= min_size:
            if newest is None or st.st_mtime > newest_mtime:
                newest = p
                newest_mtime = st.st_mtime
    return newest

def wait_for_new_pdf(download_dir: Path, since_ts: float, timeout: int, min_size: int, poll: float) -> Path | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        p = newest_pdf_since(download_dir, since_ts, min_size)
        if p and p.exists():
            s1 = p.stat().st_size
            time.sleep(max(poll, 0.5))
            s2 = p.stat().st_size
            if s2 >= min_size and s2 == s1:
                return p
        time.sleep(poll)
    return None

def attach_pdf_to_item(zot, item_key: str, pdf_path: Path) -> None:
    zot.attachment_simple([str(pdf_path)], item_key)

# -------- Main flow --------
def main():
    ap = argparse.ArgumentParser(description="Interactive: open LLM:Relevant items (no-PDF) and attach downloaded PDFs.")
    ap.add_argument("--query-tag", default=DEFAULT_QUERY_TAG, help='Tag to search (default: "LLM:Relevant")')
    ap.add_argument("--skip-tag", default=DEFAULT_SKIP_TAG, help='Tag to add when you skip an item (default: "needs-manual")')
    ap.add_argument("--downloads", default=None, help="Downloads folder path (default: auto-detect)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Seconds to wait for each download (default 300)")
    ap.add_argument("--min-pdf-size", type=int, default=DEFAULT_MIN_PDF_SIZE, help="Minimum valid PDF size (bytes)")
    ap.add_argument("--poll", type=float, default=DEFAULT_POLL, help="Polling interval seconds (default 0.3)")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N items")
    ap.add_argument("--open-only", action="store_true", help="Just open pages and prompt; do not watch or attach")
    args = ap.parse_args()

    # Env
    ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
    ZOTERO_LIBRARY_ID = os.getenv("ZOTERO_LIBRARY_ID")
    ZOTERO_LIBRARY_TYPE = os.getenv("ZOTERO_LIBRARY_TYPE") or "user"
    if not ZOTERO_API_KEY or not ZOTERO_LIBRARY_ID:
        raise SystemExit("ERROR: Set ZOTERO_API_KEY and ZOTERO_LIBRARY_ID in .env")

    download_dir = detect_downloads_dir(args.downloads)
    zot = zotero.Zotero(ZOTERO_LIBRARY_ID, ZOTERO_LIBRARY_TYPE, ZOTERO_API_KEY)

    # Fetch candidates: tag matches, no PDF yet, and not already skip-tagged
    items = zot.everything(zot.items(tag=args.query_tag))
    candidates = []
    for it in items:
        key = it["data"]["key"]
        if has_pdf_attachment(zot, key):
            continue
        if args.skip_tag and item_has_tag(it, args.skip_tag):
            continue
        candidates.append(it)

    if args.limit:
        candidates = candidates[: args.limit]

    total = len(candidates)
    print(f"[INFO] Will process {total} item(s) with tag='{args.query_tag}', no-PDF, skip-tag='{args.skip_tag or '(none)'}'.")
    print(f"[INFO] Downloads={download_dir} | timeout={args.timeout}s | min_pdf_size={args.min_pdf_size}B\n")

    processed = attached = skipped = 0

    for idx, it in enumerate(candidates, 1):
        key = it["data"]["key"]
        title = it["data"].get("title") or it["data"].get("shortTitle") or key

        print(f"[{idx}/{total}] {title}  (key={key})")
        opened = open_item_in_browser(it)
        if not opened:
            print("  No DOI or URL. (S)kip or (Q)uit?")
            cmd = input("  Enter: s to skip & tag; q to quit; anything else = skip w/o tag: ").strip().lower()
            if cmd == "q":
                break
            if cmd == "s" and args.skip_tag:
                add_tag_to_item(zot, it, args.skip_tag)
                skipped += 1
            else:
                skipped += 1
            print()
            continue

        # Ask user how to proceed
        print(f"  Opened: {opened}")
        print("  Actions: [Enter] start watching for download | (s) skip & tag | (o) open again | (q) quit")
        while True:
            cmd = input("  >> ").strip().lower()
            if cmd == "q":
                print("\n[EXIT]\n")
                print(f"[SUMMARY] processed={processed}, attached={attached}, skipped={skipped}")
                return
            if cmd == "o":
                webbrowser.open_new_tab(opened)
                continue
            if cmd == "s":
                if args.skip_tag:
                    add_tag_to_item(zot, it, args.skip_tag)
                skipped += 1
                print("  [SKIPPED]\n")
                break
            # default: start watching
            if args.open_only:
                print("  [OPEN-ONLY MODE] not watching or attaching.\n")
                break

            since_ts = time.time()
            print("  Watching Downloads for a new PDF... (Ctrl+C to abort watching for this item)")
            try:
                pdf = wait_for_new_pdf(download_dir, since_ts, args.timeout, args.min_pdf_size, args.poll)
            except KeyboardInterrupt:
                print("  [ABORT WATCH] This item not attached.")
                print("  (s) skip & tag | (o) open again | [Enter] to watch again | (n) next item")
                cmd2 = input("  >> ").strip().lower()
                if cmd2 == "s":
                    if args.skip_tag:
                        add_tag_to_item(zot, it, args.skip_tag)
                    skipped += 1
                    print("  [SKIPPED]\n")
                    break
                if cmd2 == "o":
                    webbrowser.open_new_tab(opened)
                    continue
                if cmd2 == "n":
                    print("  [NEXT]\n")
                    break
                else:
                    continue  # watch again

            if not pdf:
                print("  [TIMEOUT] No new PDF detected.")
                print("  (s) skip & tag | (o) open again | [Enter] watch again | (n) next item")
                cmd3 = input("  >> ").strip().lower()
                if cmd3 == "s":
                    if args.skip_tag:
                        add_tag_to_item(zot, it, args.skip_tag)
                    skipped += 1
                    print("  [SKIPPED]\n")
                    break
                if cmd3 == "o":
                    webbrowser.open_new_tab(opened)
                    continue
                if cmd3 == "n":
                    print("  [NEXT]\n")
                    break
                else:
                    continue  # watch again

            # Attach on success
            try:
                attach_pdf_to_item(zot, key, pdf)
                attached += 1
                print(f"  [OK] Attached {pdf.name}\n")
            except Exception as e:
                print(f"  [ATTACH FAIL] {type(e).__name__}: {e}\n")
            break

        processed += 1

    print(f"\n[SUMMARY] processed={processed}, attached={attached}, skipped={skipped}")
    print("Done.")

if __name__ == "__main__":
    main()
