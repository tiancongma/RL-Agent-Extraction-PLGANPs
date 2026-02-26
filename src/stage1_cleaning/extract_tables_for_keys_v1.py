#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
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
            key = str(obj.get("zotero_key", "")).strip()
            if not key:
                continue
            attachments = obj.get("attachments", [])
            if not isinstance(attachments, list):
                attachments = []
            for a in attachments:
                if not isinstance(a, dict):
                    continue
                out.setdefault(key, []).append(
                    {
                        "attachment_key": str(a.get("attachment_key", "")).strip(),
                        "filename": str(a.get("filename", "")).strip(),
                        "local_path": str(a.get("local_path", "")).strip(),
                        "content_type": str(a.get("content_type", "")).strip(),
                    }
                )
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
    except Exception:
        return out, "camelot_not_available"

    for flavor in ["lattice", "stream"]:
        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor=flavor)
        except Exception:
            reason = f"camelot_{flavor}_failed"
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
    return out, "no_pdf_tables_found"


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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = tables_root / zotero_key
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stable ordering for reproducibility.
    extracted_sorted = sorted(
        extracted,
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
        [x["filename"] for x in written_files if x.get("source_type") == "html_table"]
    )
    pdf_files = sorted(
        [x["filename"] for x in written_files if x.get("source_type") == "pdf_table"]
    )

    preferred_table_source = "none"
    selected_table_files: list[str] = []
    fallback_table_files: list[str] = []
    if n_tables_html_extracted > 0:
        preferred_table_source = "html"
        selected_table_files = html_files
        fallback_table_files = pdf_files
    elif n_tables_pdf_extracted > 0:
        preferred_table_source = "pdf"
        selected_table_files = pdf_files
        fallback_table_files = []

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
        "n_tables_pdf_extracted": int(n_tables_pdf_extracted),
        "total_tables": int(len(manifest_rows)),
        "html_table_reason": html_reason if chosen.html_found and n_tables_html_extracted == 0 else "",
        "pdf_table_reason": pdf_reason if chosen.pdf_found and n_tables_pdf_extracted == 0 else "",
        "preferred_table_source": preferred_table_source,
        "selected_table_files": selected_table_files,
        "fallback_table_files": fallback_table_files,
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
        total_tables += len(all_tables)
        _, idx_rows = write_tables_for_key(
            k,
            all_tables,
            chosen,
            html_reason=html_reason,
            pdf_reason=pdf_reason,
            n_tables_html_extracted=len(html_tables),
            n_tables_pdf_extracted=len(pdf_tables),
            tables_root=tables_root,
        )
        global_index_rows.extend(idx_rows)

        print(f"[key={k}] html_found={chosen.html_found} pdf_found={chosen.pdf_found}")
        print(f"[key={k}] chosen_html_path={str(chosen.html_path) if chosen.html_path else ''}")
        print(f"[key={k}] chosen_pdf_path={str(chosen.pdf_path) if chosen.pdf_path else ''}")
        print(f"[key={k}] n_tables_html_extracted={len(html_tables)} n_tables_pdf_extracted={len(pdf_tables)} total_tables={len(all_tables)}")
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
