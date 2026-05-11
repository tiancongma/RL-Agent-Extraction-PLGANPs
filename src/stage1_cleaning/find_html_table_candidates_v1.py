#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "src" / "utils" / "paths.py").exists():
            sys.path.insert(0, str(p))
            return


_bootstrap_import_paths()
from src.utils import paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find HTML table-friendly Zotero attachment candidates.")
    p.add_argument("--max-bytes", type=int, default=5 * 1024 * 1024, help="Max bytes to read from each HTML file (default 5MB).")
    return p.parse_args()


def resolve_zotero_storage_root() -> tuple[Path, list[Path]]:
    tried: list[Path] = []
    env = os.getenv("ZOTERO_STORAGE_DIR", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        tried.append(p)
        if p.exists() and p.is_dir():
            return p, tried

    userprofile = Path(os.getenv("USERPROFILE", "")).expanduser() if os.getenv("USERPROFILE") else Path.home()
    appdata = Path(os.getenv("APPDATA", "")).expanduser() if os.getenv("APPDATA") else None
    candidates = [
        userprofile / "Zotero" / "storage",
        userprofile / "Documents" / "Zotero" / "storage",
    ]
    if appdata is not None:
        candidates.append(appdata / "Zotero" / "storage")
    for c in candidates:
        tried.append(c.resolve())
        if c.exists() and c.is_dir():
            return c.resolve(), tried
    raise RuntimeError(
        "Could not locate Zotero storage root. Tried:\n"
        + "\n".join([f"- {str(x)}" for x in tried])
        + "\nSet ZOTERO_STORAGE_DIR to Zotero storage path."
    )


def load_attachment_registry() -> tuple[dict[str, str], dict[str, list[dict[str, str]]]]:
    files = [
        paths.DATA_RAW_DIR / "zotero" / "zotero_collection__goren_2025.jsonl",
        paths.DATA_RAW_DIR / "zotero" / "zotero_llm_relevant.jsonl",
        paths.DATA_RAW_DIR / "zotero" / "zotero_selected_items.jsonl",
    ]
    attachment_to_key: dict[str, str] = {}
    key_to_attachments: dict[str, list[dict[str, str]]] = {}
    for p in files:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            zkey = str(obj.get("zotero_key", "")).strip()
            if not zkey:
                continue
            attachments = obj.get("attachments", [])
            if not isinstance(attachments, list):
                continue
            for a in attachments:
                if not isinstance(a, dict):
                    continue
                ak = str(a.get("attachment_key", "")).strip()
                filename = str(a.get("filename", "")).strip()
                local_path = str(a.get("local_path", "")).strip()
                content_type = str(a.get("content_type", "")).strip()
                if ak and ak not in attachment_to_key:
                    attachment_to_key[ak] = zkey
                key_to_attachments.setdefault(zkey, []).append(
                    {
                        "attachment_key": ak,
                        "filename": filename,
                        "local_path": local_path,
                        "content_type": content_type,
                    }
                )
    return attachment_to_key, key_to_attachments


def choose_largest_html(folder: Path) -> Path | None:
    files = [p for p in folder.glob("*.html")] + [p for p in folder.glob("*.htm")]
    files = [p for p in files if p.is_file()]
    if not files:
        return None
    return sorted(files, key=lambda x: (x.stat().st_size, str(x).lower()), reverse=True)[0]


def choose_largest_path(paths_in: list[Path]) -> Path | None:
    if not paths_in:
        return None
    return sorted(paths_in, key=lambda x: (x.stat().st_size, str(x).lower()), reverse=True)[0]


def sample_html_stats(path: Path, max_bytes: int) -> tuple[bool, int, list[str], int]:
    raw = path.read_bytes()[:max_bytes]
    text = raw.decode("utf-8", errors="ignore")
    lower = text.lower()
    n_table = len(re.findall(r"<table\b", lower))
    has_table = n_table > 0
    keywords = [
        ("table", r"\btable\b"),
        ("factorial", r"\bfactorial\b"),
        ("design", r"\bdesign\b"),
        ("layout", r"\blayout\b"),
        ("coded", r"\bcoded\b"),
        ("ee", r"\bee\b"),
        ("entrapment", r"\bentrapment\b"),
        ("size", r"\bsize\b"),
        ("pdi", r"\bpdi\b"),
        ("zeta", r"\bzeta\b"),
    ]
    hit = [k for k, pat in keywords if re.search(pat, lower)]
    return has_table, n_table, hit, len(hit)


def main() -> None:
    args = parse_args()
    storage_root, tried = resolve_zotero_storage_root()
    folder_to_key, key_to_attachments = load_attachment_registry()

    rows: list[dict[str, Any]] = []
    candidates_by_key: dict[str, list[Path]] = {}

    # From storage folders directly.
    for folder in sorted([p for p in storage_root.iterdir() if p.is_dir()], key=lambda x: x.name):
        html_path = choose_largest_html(folder)
        if html_path is None:
            continue
        zkey = folder_to_key.get(folder.name, folder.name)
        candidates_by_key.setdefault(zkey, []).append(html_path.resolve())

    # From attachment metadata local paths / attachment-key folder.
    for zkey, att_list in key_to_attachments.items():
        for a in att_list:
            ak = str(a.get("attachment_key", "")).strip()
            fn = str(a.get("filename", "")).strip()
            lp = str(a.get("local_path", "")).strip()
            poss: list[Path] = []
            if lp:
                poss.append(Path(lp).expanduser())
            if ak and fn:
                poss.append(storage_root / ak / fn)
            if ak:
                adir = storage_root / ak
                if adir.exists():
                    poss.extend(list(adir.glob("*.html")))
                    poss.extend(list(adir.glob("*.htm")))
            for p in poss:
                if not p.exists() or not p.is_file():
                    continue
                if p.suffix.lower() in {".html", ".htm"}:
                    candidates_by_key.setdefault(zkey, []).append(p.resolve())

    # One best HTML per zotero_key.
    for zkey in sorted(candidates_by_key.keys()):
        uniq = sorted(set(candidates_by_key[zkey]), key=lambda x: str(x).lower())
        best = choose_largest_path(uniq)
        if best is None:
            continue
        has_table, n_table, hit, hit_count = sample_html_stats(best, max_bytes=args.max_bytes)
        rows.append(
            {
                "zotero_key": zkey,
                "html_path": str(best.resolve()),
                "file_size_mb": round(float(best.stat().st_size) / (1024.0 * 1024.0), 3),
                "has_table_tag": bool(has_table),
                "n_table_tags": int(n_table),
                "keyword_hits": "|".join(hit),
                "keyword_hit_count": int(hit_count),
            }
        )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["keyword_hit_count", "n_table_tags", "file_size_mb"], ascending=[False, False, False]).reset_index(drop=True)
    out_tsv = paths.DATA_CLEANED_DIR / "content_goren_2025" / "html_table_candidates_v1.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_tsv, sep="\t", index=False)

    print(f"zotero_storage_root={storage_root}")
    print(f"zotero_storage_tried={'; '.join([str(x) for x in tried])}")
    print(f"output_tsv={out_tsv}")
    top = out_df[out_df["has_table_tag"] == True].head(20) if not out_df.empty else pd.DataFrame()
    print("[top20_has_table_tag_true]")
    if top.empty:
        print("(empty)")
    else:
        print(top[["zotero_key", "html_path", "file_size_mb", "n_table_tags", "keyword_hit_count", "keyword_hits"]].to_string(index=False))
    print("Pick 1 zotero_key from the top 20 to add to the pilot for html_table extraction.")


if __name__ == "__main__":
    main()
