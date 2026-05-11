#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import hashlib
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import pandas as pd


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "src" / "utils" / "paths.py").exists():
            sys.path.insert(0, str(p))
            return


_bootstrap_import_paths()
from src.utils import paths  # noqa: E402


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"


@dataclass
class AttachmentChoice:
    html_path: Path | None
    pdf_path: Path | None
    html_found: bool
    pdf_found: bool


def tokenize_filename(text: str) -> set[str]:
    parts = re.findall(r"[a-z0-9]+", str(text).lower())
    stop = {"the", "and", "for", "with", "from", "into", "of", "as", "in", "on", "to", "a", "an", "by", "et", "al"}
    return {p for p in parts if len(p) >= 4 and p not in stop}


def find_related_pdf_near_html(html_path: Path) -> Path | None:
    try:
        attachment_dir = html_path.parent
        storage_root = attachment_dir.parent
        if not storage_root.exists() or not storage_root.is_dir():
            return None
        html_tokens = tokenize_filename(html_path.stem)
        if not html_tokens:
            return None
        best: Path | None = None
        best_score = 0
        for pdf in storage_root.glob("*/*.pdf"):
            if not pdf.is_file():
                continue
            pdf_tokens = tokenize_filename(pdf.stem)
            overlap = len(html_tokens.intersection(pdf_tokens))
            if overlap < 4:
                continue
            size = int(pdf.stat().st_size)
            score = overlap * 1_000_000 + size
            if score > best_score:
                best_score = score
                best = pdf.resolve()
        return best
    except Exception:
        return None


def _extract_zotero_storage_suffix(path_text: str) -> tuple[str, str]:
    """Return (attachment_key, filename) from a Windows/macOS Zotero storage path."""
    s = str(path_text or "").strip().replace("\\", "/")
    if not s:
        return "", ""
    parts = [p for p in s.split("/") if p]
    lowered = [p.lower() for p in parts]
    if "storage" in lowered:
        idx = len(lowered) - 1 - lowered[::-1].index("storage")
        if idx + 2 < len(parts):
            return parts[idx + 1], parts[-1]
    return "", parts[-1] if parts else ""


def _append_attachment_record(out: dict[str, list[dict[str, str]]], key: str, attachment_key: str, filename: str, local_path: str, content_type: str = "") -> None:
    key = str(key or "").strip()
    if not key:
        return
    rec = {
        "attachment_key": str(attachment_key or "").strip(),
        "filename": str(filename or "").strip(),
        "local_path": str(local_path or "").strip(),
        "content_type": str(content_type or "").strip(),
    }
    if not rec["attachment_key"] and rec["local_path"]:
        rec["attachment_key"], inferred_filename = _extract_zotero_storage_suffix(rec["local_path"])
        if not rec["filename"]:
            rec["filename"] = inferred_filename
    if not rec["filename"] and rec["local_path"]:
        rec["filename"] = Path(rec["local_path"].replace("\\", "/")).name
    if rec not in out.setdefault(key, []):
        out[key].append(rec)


def load_zotero_attachment_registry() -> dict[str, list[dict[str, str]]]:
    files = [
        paths.DATA_RAW_DIR / "zotero" / "zotero_collection__goren_2025.jsonl",
        paths.DATA_RAW_DIR / "zotero" / "zotero_llm_relevant.jsonl",
        paths.DATA_RAW_DIR / "zotero" / "zotero_selected_items.jsonl",
    ]
    out: dict[str, list[dict[str, str]]] = {}
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
            key = str(obj.get("zotero_key", "") or obj.get("key", "") or obj.get("paper_key", "")).strip()
            if not key:
                continue
            attachments = obj.get("attachments", [])
            if not isinstance(attachments, list):
                attachments = []
            for a in attachments:
                if not isinstance(a, dict):
                    continue
                _append_attachment_record(
                    out,
                    key,
                    attachment_key=str(a.get("attachment_key", "")).strip(),
                    filename=str(a.get("filename", "")).strip(),
                    local_path=str(a.get("local_path", "")).strip(),
                    content_type=str(a.get("content_type", "")).strip(),
                )
    manifest_path = paths.DATA_CLEANED_INDEX_DIR / "manifest_current.tsv"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                key = str(row.get("paper_key") or row.get("key") or row.get("zotero_key") or "").strip()
                for field, content_type in [("html", "text/html"), ("pdf", "application/pdf")]:
                    local_path = str(row.get(field, "") or "").strip()
                    if not local_path:
                        continue
                    attachment_key, filename = _extract_zotero_storage_suffix(local_path)
                    _append_attachment_record(out, key, attachment_key, filename, local_path, content_type=content_type)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract paper-local tables for a set of zotero keys (HTML preferred, PDF fallback).",
        epilog=(
            "Usage example:\n"
            "python src/stage1_cleaning/extract_tables_for_keys_v1.py "
            "--keys WIVUCMYG,WFDTQ4VX,UFXX9WXE,5GIF3D8W"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--keys", default="", help="Comma-separated zotero keys.")
    p.add_argument("--keys-file", default="", help="Text file with one zotero key per line.")
    p.add_argument("--run-id", default=DEFAULT_RUN_ID, help=f"Run id for debug post-check (default: {DEFAULT_RUN_ID}).")
    p.add_argument(
        "--tables-root",
        default=str(paths.DATA_CLEANED_DIR / "content_goren_2025" / "tables"),
        help="Output tables root directory, usually data/cleaned/<dataset_id>/tables.",
    )
    p.add_argument(
        "--tables-index-path",
        default=str(paths.DATA_CLEANED_DIR / "content_goren_2025" / "tables_index.tsv"),
        help="Path to consolidated tables index TSV.",
    )
    p.add_argument(
        "--skip-post-check",
        action="store_true",
        help="Skip stage5 debug post-check invocation.",
    )
    p.add_argument(
        "--no-fetch-remote-table-assets",
        action="store_true",
        help=(
            "Do not resolve source-linked http(s) full-size/download table pages. "
            "By default Stage1 fetches only directly linked source table assets, "
            "without credentials, into the paper-local tables directory and parses them."
        ),
    )
    return p.parse_args()


def parse_keys(keys_arg: str, keys_file_arg: str) -> list[str]:
    keys: list[str] = []
    if keys_file_arg.strip():
        fp = Path(keys_file_arg).resolve()
        if fp.exists():
            for line in fp.read_text(encoding="utf-8", errors="replace").splitlines():
                s = line.strip()
                if s:
                    keys.append(s)
    if keys_arg.strip():
        keys += [x.strip() for x in keys_arg.split(",") if x.strip()]
    # Stable order, dedup.
    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


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


def find_attachment_dir(storage_root: Path, zotero_key: str) -> Path | None:
    direct = storage_root / zotero_key
    if direct.exists() and direct.is_dir():
        return direct.resolve()
    # Best-effort fallback search by exact folder name.
    for p in storage_root.rglob("*"):
        if p.is_dir() and p.name == zotero_key:
            return p.resolve()
    return None


def choose_largest(paths_in: list[Path]) -> Path | None:
    if not paths_in:
        return None
    return sorted(paths_in, key=lambda x: (x.stat().st_size, str(x).lower()), reverse=True)[0]


def discover_attachments(storage_root: Path, zotero_key: str, registry: dict[str, list[dict[str, str]]]) -> AttachmentChoice:
    candidates_html: list[Path] = []
    candidates_pdf: list[Path] = []

    # Metadata-driven attachment resolution (preferred).
    for rec in registry.get(zotero_key, []):
        ak = rec.get("attachment_key", "")
        fn = rec.get("filename", "")
        lp = rec.get("local_path", "")
        poss: list[Path] = []
        if lp:
            poss.append(Path(lp).expanduser())
        if ak and fn:
            poss.append(storage_root / ak / fn)
        if ak:
            adir = storage_root / ak
            if adir.exists() and adir.is_dir():
                poss.extend(sorted(adir.glob("*")))
        for p in poss:
            if not p.exists() or not p.is_file():
                continue
            low = p.suffix.lower()
            if low in {".html", ".htm"}:
                candidates_html.append(p.resolve())
            elif low == ".pdf":
                candidates_pdf.append(p.resolve())

    # Fallback directory discovery by key folder name.
    adir = find_attachment_dir(storage_root, zotero_key)
    if adir is not None:
        candidates_html.extend(sorted([p.resolve() for p in adir.glob("*.html")] + [p.resolve() for p in adir.glob("*.htm")], key=lambda x: str(x).lower()))
        candidates_pdf.extend(sorted([p.resolve() for p in adir.glob("*.pdf")], key=lambda x: str(x).lower()))

    # Dedupe.
    html_files = sorted(set(candidates_html), key=lambda x: str(x).lower())
    pdf_files = sorted(set(candidates_pdf), key=lambda x: str(x).lower())
    chosen_html = choose_largest(html_files)
    chosen_pdf = choose_largest(pdf_files)
    if chosen_pdf is None and chosen_html is not None:
        related_pdf = find_related_pdf_near_html(chosen_html)
        if related_pdf is not None:
            chosen_pdf = related_pdf
    return AttachmentChoice(
        html_path=chosen_html,
        pdf_path=chosen_pdf,
        html_found=bool(chosen_html),
        pdf_found=bool(chosen_pdf),
    )


def clean_cell(v: Any) -> str:
    s = "" if v is None else str(v)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _table_asset_kind_from_href(href: str) -> str:
    suffix = Path(urlparse(href).path).suffix.lower().lstrip(".")
    if suffix in {"csv", "tsv", "xls", "xlsx", "html", "htm", "xml", "json"}:
        return suffix
    return "html"


def _resolve_local_href(html_path: Path, href: str) -> str:
    parsed = urlparse(href)
    if parsed.scheme in {"http", "https"}:
        return ""
    # Fragment-only anchors (for example '#t0010') point back into the current
    # source document, not to a separate source-readable table asset. Returning
    # the parent directory here causes noisy directory parse failures downstream.
    if not parsed.path:
        return ""
    if parsed.scheme == "file":
        return str(Path(parsed.path).resolve()) if parsed.path else ""
    candidate = (html_path.parent / parsed.path).resolve()
    return str(candidate) if candidate.exists() else ""


def _remote_asset_cache_path(cache_dir: Path, href: str, asset_kind: str) -> Path:
    suffix = Path(urlparse(href).path).suffix.lower()
    if suffix not in {".csv", ".tsv", ".xls", ".xlsx", ".html", ".htm"}:
        suffix = ".html" if (asset_kind or "").lower() in {"html", "htm", ""} else f".{asset_kind.lower()}"
    digest = hashlib.sha256(href.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"linked_table_asset__{digest}{suffix}"


def _fetch_remote_table_asset(href: str, asset_kind: str, cache_dir: Path, timeout_seconds: int = 30) -> tuple[str, str]:
    """Fetch a directly linked source table asset into a deterministic local cache.

    This is deliberately narrow: it only resolves http(s) links that the source
    HTML itself exposes as table/full-size/download table assets, uses no
    credentials, and stores the exact fetched page/file next to the paper's
    governed table outputs so repeated Stage1 runs can reuse the cached asset.
    """
    parsed = urlparse(href)
    if parsed.scheme not in {"http", "https"}:
        return "", "not_remote_http_asset"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = _remote_asset_cache_path(cache_dir, href, asset_kind)
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path.resolve()), "remote_cache_reused"
    try:
        req = Request(
            href,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; PLGA-Stage1-TableAssetResolver/1.0)",
                "Accept": "text/html,application/xhtml+xml,text/csv,text/tab-separated-values,application/vnd.ms-excel,*/*;q=0.8",
            },
        )
        with urlopen(req, timeout=timeout_seconds) as response:
            body = response.read()
        if not body:
            return "", "remote_fetch_empty_response"
        out_path.write_bytes(body)
        return str(out_path.resolve()), "remote_fetched"
    except Exception as e:
        return "", f"remote_fetch_failed:{type(e).__name__}"


def discover_html_table_asset_links(html_path: Path) -> list[dict[str, Any]]:
    """Discover publisher full-size/download table assets linked from source HTML.

    Stage1 owns source-readable table asset extraction: when a publisher page
    exposes a local or resolvable full-size/download table resource, Stage1 must
    preserve the locator/provenance and attempt to parse the linked resource into
    the same governed table CSV/manifest surface as inline HTML/PDF tables. S2-2
    then consumes those Stage1 table authorities for structure normalization,
    evidence construction, and drift repair; it should not be the first layer to
    open source-table links that Stage1 could read.
    """
    if html_path is None or not html_path.exists():
        return []
    html_text = html_path.read_text(encoding="utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        BeautifulSoup = None  # type: ignore
    records: list[dict[str, Any]] = []
    if BeautifulSoup is None:
        return records
    soup = BeautifulSoup(html_text, "lxml")
    for anchor in soup.find_all("a"):
        href = clean_cell(anchor.get("href", ""))
        if not href:
            continue
        link_text = clean_cell(anchor.get_text(separator=" ", strip=True))
        parsed_href = urlparse(href)
        href_path = parsed_href.path or ""
        href_suffix = Path(href_path).suffix.lower()
        is_fragment_only = not href_path and bool(parsed_href.fragment)
        explicit_table_asset_signal = any(
            token in f"{link_text} {href}".lower()
            for token in ["full size table", "download table", "view table"]
        ) or href_suffix in {".csv", ".tsv", ".xls", ".xlsx"}
        if is_fragment_only or not explicit_table_asset_signal:
            continue
        signal = f"{link_text} {href}".lower()
        caption = ""
        previous = anchor.find_previous(string=re.compile(r"\btable\s+\d+\b", re.I))
        if previous:
            parent = getattr(previous, "parent", None)
            caption = clean_cell(parent.get_text(separator=" ", strip=True) if parent is not None else str(previous))
        table_id = ""
        match = re.search(r"\btable\s+(\d+)\b", caption or link_text or href, flags=re.I)
        if match:
            table_id = f"Table {int(match.group(1))}"
        records.append(
            {
                "href_raw": href,
                "href_resolved": urljoin(html_path.as_uri(), href) if not urlparse(href).scheme else href,
                "local_path": _resolve_local_href(html_path, href),
                "link_text": link_text,
                "caption_or_title": caption,
                "table_id": table_id,
                "asset_kind": _table_asset_kind_from_href(href),
                "table_source_kind": "html_full_size_table_asset",
                "source_html_path": str(html_path.resolve()),
                "provenance_parser": "extract_tables_for_keys_v1.discover_html_table_asset_links",
            }
        )
    return records


def _read_linked_table_asset_frames(asset_path: Path, asset_kind: str) -> list[pd.DataFrame]:
    suffix = asset_path.suffix.lower().lstrip(".")
    kind = (asset_kind or suffix).lower()
    frames: list[pd.DataFrame] = []
    if kind == "tsv" or suffix == "tsv":
        frames.append(pd.read_csv(asset_path, sep="\t", dtype=str).fillna(""))
    elif kind == "csv" or suffix == "csv":
        frames.append(pd.read_csv(asset_path, dtype=str).fillna(""))
    elif kind in {"xls", "xlsx"} or suffix in {"xls", "xlsx"}:
        sheets = pd.read_excel(asset_path, sheet_name=None, dtype=str)  # type: ignore[arg-type]
        frames.extend([df.fillna("") for df in sheets.values()])
    elif kind in {"html", "htm"} or suffix in {"html", "htm"}:
        try:
            frames.extend([df.fillna("") for df in pd.read_html(str(asset_path))])
        except Exception:
            parsed, _reason = extract_tables_from_html(asset_path)
            frames.extend([rec["df"] for rec in parsed if isinstance(rec.get("df"), pd.DataFrame)])
    return frames


def extract_tables_from_html_table_asset_links(
    html_path: Path,
    *,
    asset_cache_dir: Path | None = None,
    fetch_remote_assets: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse locally available publisher-linked table assets into Stage1 tables."""
    links = discover_html_table_asset_links(html_path)
    out: list[dict[str, Any]] = []
    for link in links:
        local_path_text = str(link.get("local_path", "")).strip()
        if not local_path_text:
            href = str(link.get("href_resolved", "") or link.get("href_raw", "")).strip()
            if fetch_remote_assets and asset_cache_dir is not None:
                fetched_path, fetch_status = _fetch_remote_table_asset(href, str(link.get("asset_kind", "")), asset_cache_dir)
                link["remote_resolution_status"] = fetch_status
                if fetched_path:
                    local_path_text = fetched_path
                    link["local_path"] = fetched_path
                else:
                    link["local_extraction_status"] = "not_available_local_or_remote_fetch_failed"
                    continue
            else:
                link["local_extraction_status"] = "not_available_local_or_network_fetch_not_configured"
                continue
        asset_path = Path(local_path_text)
        if not asset_path.exists():
            link["local_extraction_status"] = "local_path_missing"
            continue
        try:
            frames = _read_linked_table_asset_frames(asset_path, str(link.get("asset_kind", "")))
        except Exception as e:
            link["local_extraction_status"] = f"parse_failed:{type(e).__name__}"
            continue
        accepted = 0
        for idx, df in enumerate(frames, start=1):
            df = df.fillna("").astype(str).apply(lambda col: col.map(clean_cell))
            if df.empty or df.shape[0] < 1 or df.shape[1] < 2:
                continue
            quality, frac_numeric, hit = compute_table_quality(df)
            rec = {
                "df": df,
                "source_type": "html_table_asset",
                "extraction_method": f"linked_{str(link.get('asset_kind', 'table_asset'))}",
                "page_number": "",
                "caption_or_title": str(link.get("caption_or_title", "")),
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "fraction_numeric_cells": round(frac_numeric, 4),
                "header_keywords_hit": hit,
                "table_quality": round(quality, 4),
                "source_table_asset_link": dict(link),
                "linked_asset_table_index": idx,
            }
            out.append(rec)
            accepted += 1
        link["local_extraction_status"] = "parsed" if accepted else "parsed_no_usable_table"
        link["local_extracted_table_count"] = accepted
    return out, links


def compute_table_quality(df: pd.DataFrame) -> tuple[float, float, list[str]]:
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    total = max(1, n_rows * n_cols)
    numeric = 0
    for _, row in df.iterrows():
        for v in row.tolist():
            sv = clean_cell(v)
            if re.search(r"^-?\d+(?:\.\d+)?$", sv):
                numeric += 1
    frac_numeric = float(numeric) / float(total)
    header = " | ".join([clean_cell(c).lower() for c in df.columns])
    keywords = [
        "ee",
        "entrapment",
        "encapsulation",
        "size",
        "pdi",
        "zeta",
        "factorial",
        "design",
        "coded",
        "x1",
        "x2",
        "x3",
    ]
    hit = [k for k in keywords if k in header]
    quality = float(n_rows) * 0.4 + float(n_cols) * 0.3 + frac_numeric * 2.0 + float(len(hit)) * 0.4
    return quality, frac_numeric, hit


def nearest_heading_text(table_tag: Any) -> str:
    # Best-effort: inspect a few previous elements for heading-like tags.
    cur = table_tag
    for _ in range(12):
        cur = cur.previous_element if hasattr(cur, "previous_element") else None
        if cur is None:
            break
        name = getattr(cur, "name", None)
        if name in {"h1", "h2", "h3", "h4"}:
            text = clean_cell(getattr(cur, "get_text", lambda **_: "")(separator=" ", strip=True))
            if text:
                return text
    return ""


def manual_table_to_df(table_tag: Any) -> pd.DataFrame:
    rows: list[list[str]] = []
    trs = table_tag.find_all("tr")
    for tr in trs:
        cells = tr.find_all(["th", "td"])
        row = [clean_cell(c.get_text(separator=" ", strip=True)) for c in cells]
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    width = max(len(r) for r in rows)
    padded = [r + ([""] * (width - len(r))) for r in rows]
    header = padded[0]
    body = padded[1:] if len(padded) > 1 else []
    return pd.DataFrame(body, columns=header)


class BasicHTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[dict[str, Any]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._in_caption = False
        self._current_table: dict[str, Any] | None = None
        self._current_row: list[str] = []
        self._cell_buf: list[str] = []
        self._caption_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t == "table":
            self._in_table = True
            self._current_table = {"caption": "", "rows": []}
            self._in_row = False
            self._in_cell = False
            self._in_caption = False
            self._current_row = []
            self._cell_buf = []
            self._caption_buf = []
        elif self._in_table and t == "caption":
            self._in_caption = True
            self._caption_buf = []
        elif self._in_table and t == "tr":
            self._in_row = True
            self._current_row = []
        elif self._in_row and t in {"th", "td"}:
            self._in_cell = True
            self._cell_buf = []
        elif self._in_cell and t == "br":
            self._cell_buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"th", "td"} and self._in_cell:
            self._in_cell = False
            self._current_row.append(clean_cell("".join(self._cell_buf)))
            self._cell_buf = []
        elif t == "tr" and self._in_row:
            self._in_row = False
            if self._current_table is not None and any(clean_cell(x) for x in self._current_row):
                self._current_table["rows"].append(self._current_row)
            self._current_row = []
        elif t == "caption" and self._in_caption:
            self._in_caption = False
            if self._current_table is not None and not self._current_table.get("caption"):
                self._current_table["caption"] = clean_cell("".join(self._caption_buf))
            self._caption_buf = []
        elif t == "table" and self._in_table:
            self._in_table = False
            if self._current_table is not None:
                self.tables.append(self._current_table)
            self._current_table = None
            self._in_row = False
            self._in_cell = False
            self._in_caption = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_buf.append(data)
        elif self._in_caption:
            self._caption_buf.append(data)


def parse_html_tables_stdlib(html_text: str) -> list[dict[str, Any]]:
    parser = BasicHTMLTableParser()
    parser.feed(html_text)
    out: list[dict[str, Any]] = []
    for t in parser.tables:
        rows = t.get("rows", [])
        if not rows:
            continue
        width = max(len(r) for r in rows)
        if width <= 0:
            continue
        padded = [list(r) + ([""] * (width - len(r))) for r in rows]
        if len(padded) >= 2:
            header = padded[0]
            body = padded[1:]
        else:
            header = [f"col_{i + 1}" for i in range(width)]
            body = padded
        df = pd.DataFrame(body, columns=[clean_cell(c) if clean_cell(c) else f"col_{i + 1}" for i, c in enumerate(header)])
        out.append({"df": df, "caption_or_title": clean_cell(str(t.get("caption", "")))})
    return out


def extract_tables_from_html(html_path: Path) -> tuple[list[dict[str, Any]], str]:
    out: list[dict[str, Any]] = []
    reason = ""
    soup = None
    tags: list[Any] = []
    try:
        from bs4 import BeautifulSoup  # type: ignore

        html_text = html_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html_text, "lxml")
        tags = list(soup.find_all("table"))
    except Exception:
        soup = None
        tags = []
    if soup is not None and not tags:
        return out, "no_html_table_tags"

    pandas_tables: list[pd.DataFrame] = []
    try:
        pandas_tables = pd.read_html(str(html_path))
        pandas_tables = [x.fillna("").astype(str) for x in pandas_tables]
    except Exception:
        pandas_tables = []
        reason = "read_html_failed"

    if not pandas_tables and soup is None:
        std_tables = parse_html_tables_stdlib(html_text=html_path.read_text(encoding="utf-8", errors="replace"))
        for rec in std_tables:
            df = rec["df"]
            if df.empty or df.shape[0] < 2 or df.shape[1] < 2:
                continue
            quality, frac_numeric, hit = compute_table_quality(df)
            out.append(
                {
                    "df": df,
                    "source_type": "html_table",
                    "extraction_method": "stdlib_html_parser",
                    "page_number": "",
                    "caption_or_title": str(rec.get("caption_or_title", "")),
                    "n_rows": int(df.shape[0]),
                    "n_cols": int(df.shape[1]),
                    "fraction_numeric_cells": round(frac_numeric, 4),
                    "header_keywords_hit": hit,
                    "table_quality": round(quality, 4),
                }
            )
        if out:
            return out, ""

    if pandas_tables and not tags:
        # bs4 may be unavailable; still keep pandas tables.
        tags = [None for _ in range(len(pandas_tables))]

    for idx, t in enumerate(tags):
        caption = ""
        if t is not None:
            cap = t.find("caption")
            if cap is not None:
                caption = clean_cell(cap.get_text(separator=" ", strip=True))
            if not caption:
                caption = nearest_heading_text(t)

        if idx < len(pandas_tables):
            df = pandas_tables[idx].copy()
            df.columns = [clean_cell(c) for c in df.columns]
            df = df.apply(lambda col: col.map(clean_cell))
        else:
            if t is None:
                continue
            df = manual_table_to_df(t)
        if df.empty or df.shape[0] < 2 or df.shape[1] < 2:
            continue
        quality, frac_numeric, hit = compute_table_quality(df)
        out.append(
            {
                "df": df,
                "source_type": "html_table",
                "extraction_method": "read_html" if idx < len(pandas_tables) else "manual_html_table",
                "page_number": "",
                "caption_or_title": caption,
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "fraction_numeric_cells": round(frac_numeric, 4),
                "header_keywords_hit": hit,
                "table_quality": round(quality, 4),
            }
        )
    if out:
        return out, ""
    if reason:
        return out, reason
    return out, "manual_parse_failed"


def extract_tables_from_pdf(pdf_path: Path) -> tuple[list[dict[str, Any]], str]:
    out: list[dict[str, Any]] = []
    reason = ""
    try:
        import camelot  # type: ignore
    except Exception as e:
        return out, f"camelot_error:{type(e).__name__}"

    for flavor in ["lattice", "stream"]:
        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor=flavor)
        except Exception as e:
            reason = f"camelot_error:{type(e).__name__}:{flavor}"
            continue
        for t in tables:
            df = t.df.fillna("").astype(str).apply(lambda col: col.map(clean_cell))
            if df.empty or df.shape[0] < 2 or df.shape[1] < 2:
                continue
            quality, frac_numeric, hit = compute_table_quality(df)
            page = ""
            try:
                page = str(getattr(t, "page", ""))
            except Exception:
                page = ""
            out.append(
                {
                    "df": df,
                    "source_type": "pdf_table",
                    "extraction_method": f"camelot_{flavor}",
                    "page_number": page,
                    "caption_or_title": "",
                    "n_rows": int(df.shape[0]),
                    "n_cols": int(df.shape[1]),
                    "fraction_numeric_cells": round(frac_numeric, 4),
                    "header_keywords_hit": hit,
                    "table_quality": round(quality, 4),
                }
            )
    if out:
        return out, ""
    if reason:
        return out, reason
    return out, "no_tables_detected"


def _normalize_pdf_reason(pdf_found: bool, n_tables_pdf_extracted: int, pdf_reason: str) -> str:
    if not pdf_found:
        return ""
    if n_tables_pdf_extracted > 0:
        return ""
    reason = (pdf_reason or "").strip()
    return reason if reason else "no_tables_detected"


def rel_to_project(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(paths.PROJECT_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(p.resolve()).replace("\\", "/")


def write_tables_for_key(
    zotero_key: str,
    extracted: list[dict[str, Any]],
    chosen: AttachmentChoice,
    html_reason: str,
    pdf_reason: str,
    n_tables_html_extracted: int,
    n_tables_pdf_extracted: int,
    tables_root: Path,
    fetch_remote_table_assets: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = tables_root / zotero_key
    out_dir.mkdir(parents=True, exist_ok=True)

    html_table_asset_links: list[dict[str, Any]] = []
    linked_asset_tables: list[dict[str, Any]] = []
    if chosen.html_path is not None:
        linked_asset_tables, html_table_asset_links = extract_tables_from_html_table_asset_links(
            chosen.html_path,
            asset_cache_dir=out_dir / "html_table_assets",
            fetch_remote_assets=fetch_remote_table_assets,
        )

    extracted_with_assets = list(extracted) + linked_asset_tables

    # Stable ordering for reproducibility.
    extracted_sorted = sorted(
        extracted_with_assets,
        key=lambda x: (
            x.get("source_type", ""),
            x.get("extraction_method", ""),
            str(x.get("page_number", "")),
            -float(x.get("table_quality", 0.0)),
        ),
    )

    manifest_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    written_files: list[dict[str, str]] = []
    for i, rec in enumerate(extracted_sorted, start=1):
        nn = f"{i:02d}"
        source_type = str(rec.get("source_type", "unknown"))
        csv_name = f"{zotero_key}__table_{nn}__{source_type}.csv"
        csv_path = out_dir / csv_name
        df: pd.DataFrame = rec["df"]  # type: ignore[assignment]
        df.to_csv(csv_path, index=False, encoding="utf-8")
        rel_csv = rel_to_project(csv_path)
        row = {
            "zotero_key": zotero_key,
            "csv_path": rel_csv,
            "source_type": source_type,
            "extraction_method": str(rec.get("extraction_method", "")),
            "page_number": str(rec.get("page_number", "")),
            "caption_or_title": str(rec.get("caption_or_title", "")),
            "n_rows": int(rec.get("n_rows", 0)),
            "n_cols": int(rec.get("n_cols", 0)),
            "fraction_numeric_cells": float(rec.get("fraction_numeric_cells", 0.0)),
            "header_keywords_hit": list(rec.get("header_keywords_hit", [])),
            "chosen_attachment_paths": {
                "html": str(chosen.html_path) if chosen.html_path is not None else "",
                "pdf": str(chosen.pdf_path) if chosen.pdf_path is not None else "",
            },
            "source_table_asset_link": dict(rec.get("source_table_asset_link", {})),
            "linked_asset_table_index": str(rec.get("linked_asset_table_index", "")),
        }
        manifest_rows.append(row)
        written_files.append(
            {
                "filename": csv_name,
                "source_type": source_type,
            }
        )
        index_rows.append(
            {
                "zotero_key": zotero_key,
                "csv_path": rel_csv,
                "source_type": source_type,
                "extraction_method": str(rec.get("extraction_method", "")),
                "caption_or_title": str(rec.get("caption_or_title", "")),
            }
        )

    html_files = sorted(
        [x["filename"] for x in written_files if x.get("source_type") in {"html_table", "html_table_asset"}]
    )
    pdf_files = sorted(
        [x["filename"] for x in written_files if x.get("source_type") == "pdf_table"]
    )

    preferred_table_source = "none"
    selected_table_files: list[str] = []
    fallback_table_files: list[str] = []
    n_tables_html_asset_extracted = len(linked_asset_tables)
    if n_tables_html_extracted > 0 or n_tables_html_asset_extracted > 0:
        preferred_table_source = "html"
        selected_table_files = html_files
        fallback_table_files = pdf_files
    elif n_tables_pdf_extracted > 0:
        preferred_table_source = "pdf"
        selected_table_files = pdf_files
        fallback_table_files = []

    selected_table_assets = [
        dict(asset)
        for asset in html_table_asset_links
        if str(asset.get("local_path", "")).strip()
    ]
    effective_html_reason = html_reason if chosen.html_found and n_tables_html_extracted == 0 else ""
    if effective_html_reason == "no_html_table_tags" and html_table_asset_links:
        effective_html_reason = (
            "html_table_assets_extracted_no_dom_table"
            if n_tables_html_asset_extracted > 0
            else "html_table_assets_preserved_no_dom_table"
        )

    manifest_path = out_dir / "tables_manifest.json"
    manifest_payload = {
        "zotero_key": zotero_key,
        "html_found": bool(chosen.html_found),
        "pdf_found": bool(chosen.pdf_found),
        "chosen_attachment_paths": {
            "html": str(chosen.html_path) if chosen.html_path is not None else "",
            "pdf": str(chosen.pdf_path) if chosen.pdf_path is not None else "",
        },
        "n_tables_html_extracted": int(n_tables_html_extracted),
        "n_tables_html_asset_extracted": int(n_tables_html_asset_extracted),
        "n_tables_pdf_extracted": int(n_tables_pdf_extracted),
        "total_tables": int(len(manifest_rows)),
        "html_table_reason": effective_html_reason,
        "pdf_table_reason": _normalize_pdf_reason(
            pdf_found=bool(chosen.pdf_found),
            n_tables_pdf_extracted=int(n_tables_pdf_extracted),
            pdf_reason=pdf_reason,
        ),
        "preferred_table_source": preferred_table_source,
        "selected_table_files": selected_table_files,
        "fallback_table_files": fallback_table_files,
        "html_table_asset_links": html_table_asset_links,
        "selected_table_assets": selected_table_assets,
        "selection_rule": "html_first_if_nonempty_else_pdf",
        "tables": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_rows, index_rows


def update_global_index(rows: list[dict[str, Any]], tables_index_path: Path) -> Path:
    p = tables_index_path
    new_df = pd.DataFrame(rows, columns=["zotero_key", "csv_path", "source_type", "extraction_method", "caption_or_title"])
    if p.exists():
        old = pd.read_csv(p, sep="\t", dtype=str).fillna("")
        out = pd.concat([old, new_df], axis=0, ignore_index=True)
    else:
        out = new_df
    out = out.drop_duplicates(subset=["zotero_key", "csv_path"], keep="last").sort_values(["zotero_key", "csv_path"]).reset_index(drop=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, sep="\t", index=False)
    return p


def main() -> None:
    args = parse_args()
    keys = parse_keys(args.keys, args.keys_file)
    if not keys:
        raise RuntimeError("No keys provided. Use --keys or --keys-file.")
    tables_root = Path(args.tables_root).resolve()
    tables_index_path = Path(args.tables_index_path).resolve()
    tables_root.mkdir(parents=True, exist_ok=True)
    tables_index_path.parent.mkdir(parents=True, exist_ok=True)

    storage_root, tried_paths = resolve_zotero_storage_root()
    attachment_registry = load_zotero_attachment_registry()
    print(f"zotero_storage_root={storage_root}")
    print(f"zotero_storage_tried={'; '.join([str(x) for x in tried_paths])}")

    total_tables = 0
    total_keys = 0
    global_index_rows: list[dict[str, Any]] = []
    keys_for_debug = []
    for k in keys:
        total_keys += 1
        keys_for_debug.append(k)
        chosen = discover_attachments(storage_root=storage_root, zotero_key=k, registry=attachment_registry)
        html_tables: list[dict[str, Any]] = []
        pdf_tables: list[dict[str, Any]] = []
        html_reason = ""
        pdf_reason = ""
        if chosen.html_path is not None:
            html_tables, html_reason = extract_tables_from_html(chosen.html_path)
        if chosen.pdf_path is not None:
            # Always run PDF path if PDF exists.
            pdf_tables, pdf_reason = extract_tables_from_pdf(chosen.pdf_path)

        all_tables = html_tables + pdf_tables
        _, idx_rows = write_tables_for_key(
            k,
            all_tables,
            chosen,
            html_reason=html_reason,
            pdf_reason=pdf_reason,
            n_tables_html_extracted=len(html_tables),
            n_tables_pdf_extracted=len(pdf_tables),
            tables_root=tables_root,
            fetch_remote_table_assets=not args.no_fetch_remote_table_assets,
        )
        global_index_rows.extend(idx_rows)
        total_tables += len(idx_rows)

        print(f"[key={k}] html_found={chosen.html_found} pdf_found={chosen.pdf_found}")
        print(f"[key={k}] chosen_html_path={str(chosen.html_path) if chosen.html_path else ''}")
        print(f"[key={k}] chosen_pdf_path={str(chosen.pdf_path) if chosen.pdf_path else ''}")
        print(
            f"[key={k}] n_tables_html_extracted={len(html_tables)} "
            f"n_tables_pdf_extracted={len(pdf_tables)} "
            f"total_tables_written={len(idx_rows)}"
        )
        ranked = sorted(all_tables, key=lambda x: float(x.get("table_quality", 0.0)), reverse=True)[:2]
        if ranked:
            for rec in ranked:
                name = rec.get("source_type", "")
                shape = f"{rec.get('n_rows',0)}x{rec.get('n_cols',0)}"
                frac = rec.get("fraction_numeric_cells", 0.0)
                cap = str(rec.get("caption_or_title", ""))
                print(f"[key={k}] top_table source={name} shape={shape} frac_numeric={frac} caption={cap[:120]}")
        else:
            print(f"[key={k}] top_table (none)")

    idx_path = update_global_index(global_index_rows, tables_index_path=tables_index_path)
    print(f"tables_index_updated={idx_path}")
    print(f"total_keys_processed={total_keys}")
    print(f"total_tables_extracted={total_tables}")

    if not args.skip_post_check:
        # Mandatory post-check for legacy invocation path.
        debug_script = paths.SRC_DIR / "stage5_benchmark" / "debug_paper_local_tables_registry_v1.py"
        debug_cmd = [
            sys.executable,
            str(debug_script),
            "--keys",
            ",".join(keys_for_debug),
            "--run-id",
            args.run_id,
        ]
        print(f"post_check_cmd={' '.join(debug_cmd)}")
        proc = subprocess.run(debug_cmd, capture_output=True, text=True)
        print(proc.stdout.strip())
        if proc.stderr.strip():
            print(proc.stderr.strip())

        debug_tsv = paths.DATA_RESULTS_DIR / args.run_id / "audit_pack" / "paper_local_tables_debug_v1.tsv"
        if debug_tsv.exists():
            dbg = pd.read_csv(debug_tsv, sep="\t", dtype=str).fillna("")
            dbg = dbg[dbg["zotero_key"].isin(keys_for_debug)].copy()
            bad = dbg[
                (dbg["reason_if_zero"].astype(str) == "tables_dir_exists_but_no_csv")
                | (dbg["total_n_csv_found"].astype(str) == "0")
            ]
            print("[post_check_for_selected_keys]")
            if bad.empty:
                print("PASS: all selected keys have total_n_csv_found > 0 and no tables_dir_exists_but_no_csv")
            else:
                print("WARN: some selected keys still have zero paper-local csv candidates")
                print(bad[["zotero_key", "total_n_csv_found", "reason_if_zero"]].to_string(index=False))
        else:
            print(f"WARN: debug TSV not found at {debug_tsv}")


if __name__ == "__main__":
    main()
