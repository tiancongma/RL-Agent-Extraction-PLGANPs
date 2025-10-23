#!/usr/bin/env python
"""
fill_missing_snapshots.py

Detect and supplement missing "snapshot-like" attachments for Zotero items
tagged `llm:relevant`. Uses .env for credentials.

Env (exact names):
  ZOTERO_LIBRARY_TYPE = user | group   (default: user)
  ZOTERO_LIBRARY_ID   = <numeric id>   (required)
  ZOTERO_API_KEY      = <token>        (required)

URL priority:
  1) parent.data.url
  2) parent.data.DOI -> https://doi.org/<DOI>
  3) child attachments: data.sourceURL, or linkMode==linked_url -> data.url

If HTML upload fails or URL is not HTML / returns 403, we fall back to a linked-URL attachment.

Usage:
  python fill_missing_snapshots.py --max-items 10 --dry-run
  python fill_missing_snapshots.py --only-key 52FJIE7T
  python fill_missing_snapshots.py --max-items 36 --link-fallback
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv, find_dotenv
from pyzotero import zotero

TAG_TO_SCAN = "llm:relevant"
ATTACHMENT_TAG = "snapshot:auto"
USER_AGENT = "Mozilla/5.0 (Snapshot-Filler)"


def _get_env_stripped(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v2 = v.strip()
    return v2 if v2 else None


def load_zotero_from_env() -> zotero.Zotero:
    # 1) cwd
    load_dotenv(override=False)
    # 2) repo root (parent of scripts/)
    repo_root_env = Path(__file__).resolve().parents[1] / ".env"
    if repo_root_env.exists():
        load_dotenv(dotenv_path=repo_root_env, override=False)
    # 3) upward discovery
    discovered = find_dotenv(usecwd=True)
    if discovered:
        load_dotenv(discovered, override=False)

    lib_type = (_get_env_stripped("ZOTERO_LIBRARY_TYPE") or "user").lower()
    lib_id = _get_env_stripped("ZOTERO_LIBRARY_ID")
    api_key = _get_env_stripped("ZOTERO_API_KEY")

    missing = []
    if not lib_id:
        missing.append("ZOTERO_LIBRARY_ID")
    if not api_key:
        missing.append("ZOTERO_API_KEY")
    if missing:
        where = [
            f"CWD .env exists? {Path('.env').resolve().exists()}",
            f"Repo .env at {repo_root_env} exists? {repo_root_env.exists()}",
            f"find_dotenv -> {discovered or 'None'}",
            f"ZOTERO_LIBRARY_TYPE={'SET:'+lib_type if lib_type else 'MISSING'}",
            f"ZOTERO_LIBRARY_ID={'SET' if lib_id else 'MISSING/EMPTY'}",
            f"ZOTERO_API_KEY={'SET' if api_key else 'MISSING/EMPTY'}",
        ]
        raise RuntimeError("Missing required env: " + ", ".join(missing) + "\n" + "\n".join(where))

    if lib_type not in {"user", "group"}:
        raise RuntimeError(f"ZOTERO_LIBRARY_TYPE must be 'user' or 'group', got: {lib_type}")

    return zotero.Zotero(library_id=lib_id, library_type=lib_type, api_key=api_key)


def list_relevant_items(zot: zotero.Zotero, start: int = 0, limit: int = 50, only_key: Optional[str] = None) -> List[Dict]:
    if only_key:
        it = zot.item(only_key)
        return [it] if it else []
    return zot.items(tag=TAG_TO_SCAN, start=start, limit=limit, sort="dateAdded", direction="desc")


def get_children(zot: zotero.Zotero, parent_key: str) -> List[Dict]:
    return zot.children(parent_key)


def is_snapshot_like(att: Dict) -> bool:
    d = att.get("data", {})
    if d.get("itemType") != "attachment":
        return False
    content_type = (d.get("contentType") or "").lower()
    link_mode = (d.get("linkMode") or "").lower()
    return content_type.startswith("text/html") and link_mode in {"imported_url", "imported_file"}


def parent_has_snapshot(children: List[Dict]) -> bool:
    return any(is_snapshot_like(ch) for ch in children)


def get_candidate_url(parent: Dict, children: List[Dict]) -> Optional[str]:
    """Return best-guess URL used by Zotero 'View Online' behavior."""
    pdata = parent.get("data", {}) or {}
    # 1) explicit URL
    u = (pdata.get("url") or "").strip()
    if u:
        return u
    # 2) DOI -> doi resolver
    doi = (pdata.get("DOI") or "").strip()
    if doi:
        return f"https://doi.org/{doi}"
    # 3) children: sourceURL or linked_url.url
    for ch in children:
        d = ch.get("data", {}) or {}
        if d.get("itemType") != "attachment":
            continue
        src = (d.get("sourceURL") or "").strip()
        if src:
            return src
        if (d.get("linkMode") or "").lower() == "linked_url":
            lu = (d.get("url") or "").strip()
            if lu:
                return lu
    return None


def sanitize_title(title: str) -> str:
    t = title.strip() if title else "Webpage snapshot"
    for bad in r'\/:*?"<>|':
        t = t.replace(bad, "_")
    return t[:120]


def fetch_url(url: str, timeout: int = 25) -> Tuple[Optional[bytes], Optional[str], Optional[int]]:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        status = r.status_code
        if status >= 400:
            return None, None, status
        ctype = (r.headers.get("Content-Type") or "").lower()
        return r.content, ctype, status
    except requests.RequestException:
        return None, None, None


def _tag_created_items(zot: zotero.Zotero, created) -> None:
    # Normalize return list
    if isinstance(created, dict) and "successful" in created:
        items = created.get("successful") or []
    elif isinstance(created, list):
        items = created
    elif created:
        items = [created]
    else:
        items = []

    for att in items:
        if isinstance(att, dict) and "key" in att and "data" not in att:
            try:
                att = zot.item(att["key"])
            except Exception:
                continue
        if isinstance(att, dict):
            data = att.get("data", {}) or {}
            tags = data.get("tags", []) or []
            tags.append({"tag": ATTACHMENT_TAG})
            data["tags"] = tags
            att["data"] = data
            try:
                zot.update_item(att)
            except Exception:
                pass


def upload_html_as_attachment(zot: zotero.Zotero, parent_key: str, filename: Path, title: str) -> bool:
    """
    Try multiple pyzotero-compatible ways to upload a local HTML file.
    Return True on success, False on failure (so caller can fall back to linked_url).
    """
    file_path_str = str(filename)
    base_name = os.path.basename(file_path_str)

    # Strategy 1: list of string paths
    try:
        created = zot.attachment_simple([file_path_str], parent_key, contentType="text/html", title=title or base_name)
        _tag_created_items(zot, created)
        return True
    except Exception:
        pass

    # Strategy 2: classic dict with 'filename'
    try:
        files = [{"filename": file_path_str, "contentType": "text/html", "title": title or base_name}]
        created = zot.attachment_simple(files, parent_key)
        _tag_created_items(zot, created)
        return True
    except Exception:
        pass

    # Strategy 3: dict with 'path'
    try:
        files = [{"path": file_path_str, "contentType": "text/html", "title": title or base_name}]
        created = zot.attachment_simple(files, parent_key)
        _tag_created_items(zot, created)
        return True
    except Exception:
        pass

    return False


def create_linked_url_attachment(zot: zotero.Zotero, parent_key: str, url: str, title: str = "Webpage (link)") -> None:
    """Create a lightweight 'linked_url' attachment (no storage used)."""
    item = {
        "itemType": "attachment",
        "parentItem": parent_key,
        "linkMode": "linked_url",
        "title": title,
        "url": url,
        "contentType": "",
        "tags": [{"tag": ATTACHMENT_TAG}],
    }
    try:
        _ = zot.create_items([item])
    except Exception:
        # best-effort; swallow errors to keep batch running
        pass


def main():
    ap = argparse.ArgumentParser(description="Detect and fill missing HTML snapshots for tagged Zotero items.")
    ap.add_argument("--max-items", type=int, default=50, help="Max parent items to process this run.")
    ap.add_argument("--batch-size", type=int, default=25, help="API page size when listing items.")
    ap.add_argument("--dry-run", action="store_true", help="Do not upload; just report actions.")
    ap.add_argument("--only-key", type=str, default=None, help="Process a single parent item key (e.g., 52FJIE7T).")
    ap.add_argument(
        "--link-fallback",
        action="store_true",
        dest="link_fallback",
        help="Create linked-URL attachment if content is not HTML."
    )
    args = ap.parse_args()

    zot = load_zotero_from_env()
    processed = supplemented = skipped_existing = skipped_no_url = failed = 0
    start = 0
    batch_size = max(1, min(args.batch_size, 100))
    max_total = args.max_items

    while processed < max_total:
        remaining = max_total - processed
        limit = min(batch_size, remaining)
        items = list_relevant_items(zot, start=start, limit=limit, only_key=args.only_key)
        if not items:
            break

        for it in items:
            key = it["key"]
            data = it.get("data", {}) or {}
            title = data.get("title") or data.get("shortTitle") or "(untitled)"

            try:
                kids = get_children(zot, key)
            except Exception as e:
                print(f"[WARN] Failed to fetch children for {key}: {e}")
                failed += 1
                processed += 1
                continue

            if parent_has_snapshot(kids):
                print(f"[SKIP] {key} (snapshot exists) title={title[:80]}")
                skipped_existing += 1
                processed += 1
                continue

            candidate = get_candidate_url(it, kids)
            if not candidate:
                print(f"[SKIP] {key} (no URL/DOI/sourceURL/linked_url) title={title[:80]}")
                skipped_no_url += 1
                processed += 1
                continue

            print(f"[FILL] {key} -> fetching {candidate}")

            if args.dry_run:
                processed += 1
                continue

            content, ctype, status = fetch_url(candidate)

            # 403 或非 HTML：优先/自动兜底为 linked_url（即使没有 --link-fallback 也执行）
            if status == 403 or (ctype is not None and not ctype.startswith("text/html")):
                create_linked_url_attachment(zot, key, candidate, title="Webpage (link)")
                supplemented += 1
                print(f"[OK] {key} fallback to linked URL (status={status}, ctype={ctype}).")
                processed += 1
                continue

            if content is None or ctype is None:
                # 网络错误等，也走链接兜底（若没给出 --link-fallback，也兜）
                create_linked_url_attachment(zot, key, candidate, title="Webpage (link)")
                supplemented += 1
                print(f"[OK] {key} fallback to linked URL (fetch failed).")
                processed += 1
                continue

            # HTML 分支：尝试真正上传快照
            try:
                safe_title = sanitize_title(title)
                with tempfile.TemporaryDirectory() as td:
                    fn = Path(td) / f"{safe_title}.html"
                    fn.write_bytes(content)
                    ok = upload_html_as_attachment(zot, parent_key=key, filename=fn, title="Webpage snapshot")
                if ok:
                    supplemented += 1
                    print(f"[OK] {key} uploaded HTML snapshot.")
                else:
                    # 上传仍失败：兜底链接
                    create_linked_url_attachment(zot, key, candidate, title="Webpage (link)")
                    supplemented += 1
                    print(f"[OK] {key} snapshot upload failed; fallback to linked URL.")
            except Exception as e:
                failed += 1
                print(f"[ERR] {key} failed to process: {e}")

            processed += 1

        if args.only_key:
            break
        start += limit

    stats = {
        "processed": processed,
        "supplemented": supplemented,
        "skipped_existing": skipped_existing,
        "skipped_no_url": skipped_no_url,
        "failed": failed,
        "dry_run": bool(args.dry_run),
        "tag_scanned": TAG_TO_SCAN,
    }
    print("Stats:", stats)


if __name__ == "__main__":
    main()
