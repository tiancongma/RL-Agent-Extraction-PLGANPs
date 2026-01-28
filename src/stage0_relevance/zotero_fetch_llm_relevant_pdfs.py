#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Attach PDFs to existing Zotero items (no duplicates) for items tagged with a specific tag,
limited to those that currently have no PDF but do have a DOI.

Priority:
  1) Unpaywall best OA PDF
  2) DOI content negotiation (Accept: application/pdf) as a fallback

What it does:
  - Finds items with tag (default: "LLM:Relevant")
  - Skips items that already have any PDF attachment
  - For each remaining item with a DOI:
      * try Unpaywall → PDF
      * else try DOI content negotiation → PDF or .pdf link scraped from HTML
      * download to temp folder
      * attach directly to the existing item via Zotero API (no new parent item)
  - Optionally tags failures with a label (e.g., "needs-manual")

Environment variables (or .env if present):
  ZOTERO_API_KEY       (required)
  ZOTERO_LIBRARY_ID    (required; numeric)
  ZOTERO_LIBRARY_TYPE  (optional; default "user"; use "group" for group libraries)
  UNPAYWALL_EMAIL      (recommended; Unpaywall requires an email)

CLI examples:
  python zotero_fetch_llm_relevant_pdfs.py --limit 10
  python zotero_fetch_llm_relevant_pdfs.py --tag "LLM:Relevant" --fail-tag "needs-manual"
"""

import os
import re
import time
import tempfile
import argparse
from urllib.parse import quote

import requests
from tqdm import tqdm
from pyzotero import zotero

try:
    # Load .env if python-dotenv is installed (safe no-op otherwise)
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ------------------ Defaults & Constants ------------------

DEFAULT_TAG = "LLM:Relevant"
REQUEST_TIMEOUT = 25
DEFAULT_SLEEP = 0.6               # seconds between items (mild throttling)
DEFAULT_MIN_PDF_SIZE = 50 * 1024  # <50 KB considered invalid PDF
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)

# ------------------ Helpers ------------------

def has_pdf_attachment(zot, item_key):
    """Return True if the item already has any PDF attachment."""
    try:
        for child in zot.children(item_key):
            data = child.get("data", {})
            if data.get("itemType") == "attachment":
                ctype = (data.get("contentType") or "").lower()
                if ctype == "application/pdf":
                    return True
        return False
    except Exception:
        return False

def extract_doi(item):
    """Extract DOI from data.DOI or from the Extra field."""
    data = item.get("data", {})
    doi = (data.get("DOI") or "").strip()
    if doi and DOI_RE.match(doi):
        return doi

    extra = data.get("extra") or ""
    m = re.search(r"\bDOI:\s*(10\.\S+)", extra, re.IGNORECASE)
    if m and DOI_RE.match(m.group(1)):
        return m.group(1)

    return None

def unpaywall_pdf_url(doi, email):
    """Return the best OA PDF URL from Unpaywall, if available."""
    if not email:
        return None
    api = f"https://api.unpaywall.org/v2/{quote(doi)}?email={quote(email)}"
    r = requests.get(api, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        return None
    js = r.json()
    loc = js.get("best_oa_location") or {}
    return loc.get("url_for_pdf") or loc.get("url")

def doi_content_negotiation_pdf(doi):
    """
    Try to get a direct PDF URL via DOI content negotiation or by scraping a .pdf link
    from the landing HTML if negotiation returns HTML.
    """
    headers = {"Accept": "application/pdf, application/*;q=0.9, */*;q=0.8"}
    r = requests.get(f"https://doi.org/{doi}", headers=headers,
                     timeout=REQUEST_TIMEOUT, allow_redirects=True)
    ctype = (r.headers.get("Content-Type", "") or "").lower()
    if r.status_code == 200 and (ctype.startswith("application/pdf") or r.url.lower().endswith(".pdf")):
        return r.url

    if "text/html" in ctype and r.text:
        # Heuristic: find a .pdf link in the HTML
        pdf_links = re.findall(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', r.text, flags=re.IGNORECASE)
        if pdf_links:
            link = pdf_links[0]
            if link.startswith(("http://", "https://")):
                return link
    return None

def safe_filename(s, ext=".pdf"):
    s = re.sub(r"[^\w\-.]+", "_", s).strip("._")
    if not s.lower().endswith(".pdf"):
        s += ext
    return s

def download_pdf(url, out_dir, min_size=DEFAULT_MIN_PDF_SIZE):
    """Download a PDF to out_dir; return the local path or None."""
    r = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True)
    if r.status_code != 200:
        return None

    # Guess file name from Content-Disposition or URL
    fname = None
    cd = r.headers.get("Content-Disposition", "")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
    if m:
        fname = m.group(1)
    else:
        fname = url.split("/")[-1].split("?")[0] or "download.pdf"
    fname = safe_filename(fname)

    path = os.path.join(out_dir, fname)
    size = 0
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)
                size += len(chunk)

    if size < min_size:
        try:
            os.remove(path)
        except Exception:
            pass
        return None
    return path

def attach_pdf_to_item(zot, parent_key, pdf_path):
    """Attach a local PDF file to the existing Zotero item (no new parent created)."""
    zot.attachment_simple([pdf_path], parent_key)

def add_tag_to_item(zot, item, tag_text):
    """Append a tag to the item (avoid duplicates) and update the item."""
    data = item.get("data", {})
    tags = data.get("tags", []) or []
    if not any((t.get("tag") or "").strip().lower() == tag_text.strip().lower() for t in tags):
        tags.append({"tag": tag_text})
        upd = {"key": data["key"], "version": data["version"], "data": data | {"tags": tags}}
        zot.update_item(upd)

# ------------------ Main ------------------

def main():
    parser = argparse.ArgumentParser(description="Attach PDFs to Zotero items (tag-filtered, DOI-based).")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Zotero tag to process (default: LLM:Relevant)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N items (small-batch test)")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Sleep seconds between items")
    parser.add_argument("--min-pdf-size", type=int, default=DEFAULT_MIN_PDF_SIZE, help="Minimum valid PDF size in bytes")
    parser.add_argument("--fail-tag", default=None, help="If set, tag items that failed to get a PDF (e.g., needs-manual)")
    parser.add_argument("--domain-blacklist", default="", help="Comma-separated domains to skip (e.g., wiley.com,ingentaconnect.com)")
    args = parser.parse_args()

    # Env / .env
    ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
    ZOTERO_LIBRARY_ID = os.getenv("ZOTERO_LIBRARY_ID")
    ZOTERO_LIBRARY_TYPE = os.getenv("ZOTERO_LIBRARY_TYPE") or "user"
    UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL")

    if not ZOTERO_API_KEY or not ZOTERO_LIBRARY_ID:
        raise SystemExit("ERROR: Please set ZOTERO_API_KEY and ZOTERO_LIBRARY_ID (and optionally ZOTERO_LIBRARY_TYPE)")

    # Use locals instead of mutating globals
    sleep_between = args.sleep
    min_pdf_size = args.min_pdf_size
    domain_blacklist = [d.strip().lower() for d in args.domain_blacklist.split(",") if d.strip()]

    zot = zotero.Zotero(ZOTERO_LIBRARY_ID, ZOTERO_LIBRARY_TYPE, ZOTERO_API_KEY)

    # Fetch items with the given tag
    items = zot.everything(zot.items(tag=args.tag))
    if args.limit:
        items = items[:args.limit]

    tdir = tempfile.mkdtemp(prefix="zpdf_")
    print(f"[INFO] Temp dir: {tdir}")
    print(f"[INFO] Items fetched by tag '{args.tag}': {len(items)} (limit={args.limit or 'none'})")

    processed = 0
    attached = 0
    skipped = 0
    failed = 0

    for it in tqdm(items, desc="Processing"):
        key = it["data"]["key"]
        title = it["data"].get("title") or it["data"].get("shortTitle") or key

        # Skip if any PDF already attached
        if has_pdf_attachment(zot, key):
            skipped += 1
            continue

        doi = extract_doi(it)
        if not doi:
            skipped += 1
            continue

        # Find candidate PDF URL
        pdf_url = None

        # 1) Unpaywall
        try:
            pdf_url = unpaywall_pdf_url(doi, UNPAYWALL_EMAIL)
        except Exception:
            pdf_url = None

        # 2) DOI content negotiation
        if not pdf_url:
            try:
                pdf_url = doi_content_negotiation_pdf(doi)
            except Exception:
                pdf_url = None

        # Domain blacklist
        if pdf_url and domain_blacklist and any(db in pdf_url.lower() for db in domain_blacklist):
            pdf_url = None

        if not pdf_url:
            failed += 1
            if args.fail_tag:
                try:
                    add_tag_to_item(zot, it, args.fail_tag)
                except Exception:
                    pass
            time.sleep(sleep_between)
            continue

        # Download
        try:
            pdf_path = download_pdf(pdf_url, tdir, min_size=min_pdf_size)
        except Exception:
            pdf_path = None

        if not pdf_path:
            failed += 1
            if args.fail_tag:
                try:
                    add_tag_to_item(zot, it, args.fail_tag)
                except Exception:
                    pass
            time.sleep(sleep_between)
            continue

        # Attach
        try:
            attach_pdf_to_item(zot, key, pdf_path)
            attached += 1
        except Exception as e:
            print(f"[ATTACH FAILED] {title} :: {e}")
            failed += 1
            if args.fail_tag:
                try:
                    add_tag_to_item(zot, it, args.fail_tag)
                except Exception:
                    pass
        finally:
            processed += 1
            time.sleep(sleep_between)

    print(f"\n[SUMMARY] processed={processed}, attached={attached}, skipped={skipped}, failed={failed}")
    print("[NOTE] Temporary files are kept at:", tdir)
    print("       You can remove that folder when done.")
    if not UNPAYWALL_EMAIL:
        print("[WARN] UNPAYWALL_EMAIL not set; Unpaywall step was skipped. Set it for higher success rates.")

if __name__ == "__main__":
    main()
