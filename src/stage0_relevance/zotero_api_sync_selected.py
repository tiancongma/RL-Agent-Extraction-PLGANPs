#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zotero_api_sync_selected.py

Sync a *tag-selected* set of Zotero items via Zotero Web API into a local raw JSONL.

This script is designed for the pipeline:
Step 0 (raw build): Zotero API -> data/raw/zotero/zotero_selected_items.jsonl
Step 1 (manifest):  zotero_selected_items.jsonl -> data/cleaned/index/manifest_current.tsv

Key features
- Tag-based selection (e.g., "LLM:Relevant")
- Incremental sync using Zotero library version ("since")
- Attachment inspection (PDF/HTML) with local path resolution
- No hard-coded repo paths (defaults via src/utils/paths.py)

Local path resolution for attachments
- Zotero stores imported files under: <ZOTERO_STORAGE_DIR>/<attachmentKey>/<filename>
- IMPORTANT: the folder name is the *attachment item key* (8 chars), not the parent item key.
- Set storage root via:
  - CLI: --storage-root
  - OR env var: ZOTERO_STORAGE_DIR

Required environment variables (Zotero API)
- ZOTERO_LIBRARY_TYPE  (e.g., "user" or "group")
- ZOTERO_LIBRARY_ID
- ZOTERO_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyzotero import zotero


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "src" / "utils" / "paths.py").exists():
            sys.path.insert(0, str(p))
            return


_bootstrap_import_paths()
from src.utils import paths  # noqa: E402


# -----------------------------
# Data models (JSON-serializable)
# -----------------------------

@dataclass
class AttachmentInfo:
    attachment_key: str
    filename: str
    content_type: str
    link_mode: str
    local_path: Optional[str]
    url: Optional[str]


@dataclass
class ItemRecord:
    zotero_key: str
    title: str
    year: str
    doi: str
    tags: List[str]
    paths: Dict[str, Any]  # pdf/html/text/sections_json/tables_csv placeholders for downstream
    attachments: List[AttachmentInfo]
    status: str
    message: str


# -----------------------------
# Helpers
# -----------------------------

def read_env_required(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def read_last_version(last_version_file: Path) -> Optional[int]:
    if not last_version_file.exists():
        return None
    txt = last_version_file.read_text(encoding="utf-8", errors="replace").strip()
    if not txt:
        return None
    try:
        return int(txt)
    except ValueError:
        return None


def write_last_version(last_version_file: Path, version: int) -> None:
    last_version_file.parent.mkdir(parents=True, exist_ok=True)
    last_version_file.write_text(str(version), encoding="utf-8")


def load_existing_jsonl(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load existing JSONL into a dict keyed by zotero_key.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return out
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            zk = str(rec.get("zotero_key", "")).strip()
            if zk:
                out[zk] = rec
    return out


def write_jsonl(jsonl_path: Path, records: Dict[str, Dict[str, Any]]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    # stable-ish order for diffs
    keys = sorted(records.keys())
    with jsonl_path.open("w", encoding="utf-8") as f:
        for k in keys:
            f.write(json.dumps(records[k], ensure_ascii=False) + "\n")


def pick_title(item: Dict[str, Any]) -> str:
    data = item.get("data", {}) or {}
    t = data.get("title") or ""
    return str(t).strip()


def pick_doi(item: Dict[str, Any]) -> str:
    data = item.get("data", {}) or {}
    doi = data.get("DOI") or data.get("doi") or ""
    return str(doi).strip()


def pick_year(item: Dict[str, Any]) -> str:
    data = item.get("data", {}) or {}
    date = str(data.get("date") or data.get("year") or "").strip()
    # keep as-is; do not parse
    return date


def pick_tags(item: Dict[str, Any]) -> List[str]:
    data = item.get("data", {}) or {}
    tags = data.get("tags") or []
    out = []
    for t in tags:
        # zotero returns [{"tag":"..."}]
        if isinstance(t, dict) and "tag" in t:
            out.append(str(t["tag"]))
        else:
            out.append(str(t))
    return out


def resolve_attachment_local_path(
    storage_root: Path,
    attachment_item_key: str,
    filename: str,
) -> Optional[str]:
    """
    Resolve Zotero storage path for imported attachments:
    <storage_root>/<attachmentKey>/<filename>
    """
    if not attachment_item_key or not filename:
        return None
    p = storage_root / attachment_item_key / filename
    return str(p) if p.exists() else None


def classify_attachment(att: Dict[str, Any]) -> Tuple[str, str, str, str, Optional[str]]:
    """
    Return (attachment_key, filename, content_type, link_mode, url)
    """
    k = str(att.get("key", "")).strip()
    data = att.get("data", {}) or {}
    fn = str(data.get("filename") or data.get("title") or "").strip()
    ct = str(data.get("contentType") or "").strip()
    lm = str(data.get("linkMode") or "").strip()
    url = data.get("url")
    return k, fn, ct, lm, str(url).strip() if url else None


def build_item_record(
    parent_item: Dict[str, Any],
    child_attachments: List[Dict[str, Any]],
    tag: str,
    storage_root: Optional[Path],
) -> ItemRecord:
    zk = str(parent_item.get("key", "")).strip()
    title = pick_title(parent_item)
    doi = pick_doi(parent_item)
    year = pick_year(parent_item)
    tags = pick_tags(parent_item)

    # Paths placeholders for downstream
    paths_payload: Dict[str, Any] = {
        "pdf": None,
        "html": None,
        "text": None,
        "sections_json": None,
        "tables_csv": [],
    }

    atts: List[AttachmentInfo] = []
    pdf_candidates: List[str] = []
    html_candidates: List[str] = []

    for att in child_attachments:
        ak, fn, ct, lm, url = classify_attachment(att)

        local_path = None
        # Only attempt local resolution when user tells us the Zotero storage root
        if storage_root is not None and lm:
            # linkMode numeric codes vary; we treat any non-empty as "may be imported"
            # For imported files, Zotero uses storage/<attachmentKey>/<filename>
            local_path = resolve_attachment_local_path(storage_root, ak, fn)

        atts.append(
            AttachmentInfo(
                attachment_key=ak,
                filename=fn,
                content_type=ct,
                link_mode=lm,
                local_path=local_path,
                url=url,
            )
        )

        # collect candidates by type
        if local_path:
            if ct.lower() == "application/pdf" or fn.lower().endswith(".pdf"):
                pdf_candidates.append(local_path)
            if fn.lower().endswith(".html") or fn.lower().endswith(".htm"):
                html_candidates.append(local_path)

    if html_candidates:
        paths_payload["html"] = html_candidates[0]
    if pdf_candidates:
        paths_payload["pdf"] = pdf_candidates[0]

    if paths_payload["pdf"] or paths_payload["html"]:
        status = "HAS_LOCAL_FULLTEXT"
        msg = "local attachment(s) found"
    else:
        status = "NO_LOCAL_FULLTEXT"
        msg = "no local html/pdf found"

    # Ensure the selection tag is visible in each record (helpful for debugging)
    if tag and tag not in tags:
        tags.append(tag)

    return ItemRecord(
        zotero_key=zk,
        title=title,
        year=year,
        doi=doi,
        tags=tags,
        paths=paths_payload,
        attachments=atts,
        status=status,
        message=msg,
    )


# -----------------------------
# Zotero API sync
# -----------------------------

def zotero_client_from_env() -> zotero.Zotero:
    lib_type = read_env_required("ZOTERO_LIBRARY_TYPE")
    lib_id = read_env_required("ZOTERO_LIBRARY_ID")
    api_key = read_env_required("ZOTERO_API_KEY")
    return zotero.Zotero(lib_id, lib_type, api_key)


def get_library_version(z: zotero.Zotero) -> int:
    """
    Get current library version with a lightweight request.

    pyzotero compatibility note:
    - Some versions expose `last_modified_version` as an attribute.
    - Other versions expose it as a method.
    We support both.
    """
    # One cheap request; pyzotero sets last modified version after calls
    _ = z.items(limit=1)

    v = getattr(z, "last_modified_version", None)

    # Some pyzotero versions implement this as a method
    if callable(v):
        v = v()

    # Fallback: try common method name explicitly
    if v is None and hasattr(z, "last_modified_version") and callable(getattr(z, "last_modified_version")):
        v = getattr(z, "last_modified_version")()

    if v is None:
        raise RuntimeError("Could not determine Zotero library version (last_modified_version missing).")

    try:
        return int(v)
    except Exception:
        return int(str(v).strip())


def fetch_changed_items_by_tag(z: zotero.Zotero, tag: str, since: Optional[int]) -> List[Dict[str, Any]]:
    """
    Fetch top-level items with a given tag, optionally since a library version.
    """
    kwargs = {"tag": tag}
    if since is not None:
        kwargs["since"] = since
    items: List[Dict[str, Any]] = []
    start = 0
    limit = 100
    while True:
        batch = z.items(start=start, limit=limit, **kwargs)
        if not batch:
            break
        items.extend(batch)
        if len(batch) < limit:
            break
        start += limit
    return items


def fetch_deleted_keys(z: zotero.Zotero, since: int) -> List[str]:
    """
    Fetch deleted item keys since a library version.
    """
    deleted = z.deleted(since=since)  # returns dict with 'items' and 'collections'
    keys = []
    if isinstance(deleted, dict):
        for k in deleted.get("items", []) or []:
            keys.append(str(k))
    return keys


def fetch_child_attachments(z: zotero.Zotero, parent_key: str) -> List[Dict[str, Any]]:
    """
    Fetch attachment children for a parent item.
    """
    # children() returns all children items; filter to attachments
    children = z.children(parent_key)
    out: List[Dict[str, Any]] = []
    for c in children:
        data = c.get("data", {}) or {}
        if data.get("itemType") == "attachment":
            out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync Zotero tag-selected items to raw JSONL (incremental).")
    ap.add_argument("--tag", default="LLM:Relevant", help='Zotero tag to select (default: "LLM:Relevant").')

    ap.add_argument(
        "--out-jsonl",
        type=Path,
        default=(paths.DATA_RAW_DIR / "zotero" / "zotero_selected_items.jsonl"),
        help="Output JSONL path (default via paths.py).",
    )
    ap.add_argument(
        "--last-version-file",
        type=Path,
        default=(paths.DATA_RAW_DIR / "zotero" / "last_version.txt"),
        help="File storing last synced Zotero library version (default via paths.py).",
    )

    ap.add_argument(
        "--storage-root",
        type=Path,
        default=None,
        help="Zotero storage root directory (e.g., .../Zotero/storage). If omitted, uses env ZOTERO_STORAGE_DIR.",
    )

    ap.add_argument("--full", action="store_true", help="Ignore last_version and do a full refresh.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")

    args = ap.parse_args()

    # Storage root resolution
    storage_root: Optional[Path] = args.storage_root
    if storage_root is None:
        env_root = os.getenv("ZOTERO_STORAGE_DIR", "").strip()
        if env_root:
            storage_root = Path(env_root)
    if storage_root is not None and not storage_root.exists():
        raise FileNotFoundError(f"storage root not found: {storage_root}")

    z = zotero_client_from_env()

    current_version = get_library_version(z)

    since = None
    if not args.full:
        since = read_last_version(args.last_version_file)

    if args.verbose:
        print(f"[INFO] tag={args.tag}")
        print(f"[INFO] current_library_version={current_version}")
        print(f"[INFO] since={since}")
        if storage_root:
            print(f"[INFO] storage_root={storage_root}")
        else:
            print("[WARN] storage root not set. Local file paths may remain null. "
                  "Set --storage-root or env ZOTERO_STORAGE_DIR.")

    existing = load_existing_jsonl(args.out_jsonl)

    # Handle deletions (only meaningful when since is known)
    if since is not None:
        deleted_keys = fetch_deleted_keys(z, since)
        for dk in deleted_keys:
            # deleted could include attachments too; we only store parent items by key
            if dk in existing:
                existing.pop(dk, None)
        if args.verbose and deleted_keys:
            print(f"[INFO] deleted_items_seen={len(deleted_keys)} (removed from local index if present)")

    changed_items = fetch_changed_items_by_tag(z, tag=args.tag, since=None if args.full else since)

    if args.verbose:
        print(f"[INFO] fetched_items={len(changed_items)}")

    updated = 0
    has_fulltext = 0

    for it in changed_items:
        data = it.get("data", {}) or {}
        # skip notes/attachments returned in top-level items (we want parent bibliographic items)
        if data.get("itemType") == "attachment":
            continue

        parent_key = str(it.get("key", "")).strip()
        if not parent_key:
            continue

        children = fetch_child_attachments(z, parent_key)
        rec = build_item_record(it, children, tag=args.tag, storage_root=storage_root)
        existing[parent_key] = json.loads(json.dumps(asdict(rec), default=str))
        updated += 1
        if rec.paths.get("pdf") or rec.paths.get("html"):
            has_fulltext += 1

    write_jsonl(args.out_jsonl, existing)
    write_last_version(args.last_version_file, current_version)

    print(f"[OK] wrote: {args.out_jsonl}")
    print(f"[OK] updated_items={updated} | total_indexed={len(existing)} | has_local_fulltext={has_fulltext}")
    print(f"[OK] saved library version -> {args.last_version_file}")


if __name__ == "__main__":
    main()
