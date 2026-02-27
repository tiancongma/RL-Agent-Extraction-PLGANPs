#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class TextEvidence:
    evidence_source_type: str
    evidence_pointer_raw: str
    evidence_text: str
    evidence_context_before: str
    evidence_context_after: str
    evidence_block_id: str
    evidence_span_id: str
    evidence_span_start: str
    evidence_span_end: str
    evidence_section: str


@dataclass
class TableEvidence:
    table_csv_path: str
    table_filename: str
    rejected_table_filename: str
    table_title_or_caption: str
    table_match_score: float
    table_row_text: str
    table_cell_text: str
    doe_signature: str
    top5_candidates: list[str]
    top5_scores: list[float]
    match_reason: str
    paper_local_candidate_count: int
    ownership_check_passed: bool
    ownership_check_reason: str
    chosen_table_rejected: bool
    table_evidence_missing_reason: str


def short_text(v: Any, limit: int) -> str:
    s = "" if v is None else str(v)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


class AuditEvidenceResolverV1:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.key2txt_map: dict[str, Path] = {}
        self.text_cache: dict[str, str] = {}
        self.table_file_index: dict[str, list[Path]] = {}
        self.table_meta: list[dict[str, Any]] = []
        self.table_paths_by_key: dict[str, list[Path]] = {}
        self._load_key2txt_maps()
        self._load_zotero_table_metadata()
        self._build_table_index()

    def _read_key2txt_no_header(self, path: Path) -> dict[str, Path]:
        out: dict[str, Path] = {}
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    k = parts[0].strip()
                    p = parts[1].strip()
                    if not k or not p:
                        continue
                    full = (self.project_root / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
                    out[k] = full
        except Exception:
            return {}
        return out

    def _read_key2txt_with_header(self, path: Path) -> dict[str, Path]:
        out: dict[str, Path] = {}
        try:
            df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        except Exception:
            return {}
        if "key" not in df.columns or "txt_path" not in df.columns:
            return {}
        base = path.parent
        for _, r in df.iterrows():
            k = str(r.get("key", "")).strip()
            rel = str(r.get("txt_path", "")).strip()
            if not k or not rel:
                continue
            full = (base / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
            out[k] = full
        return out

    def _load_key2txt_maps(self) -> None:
        # Priority: goren content > general content > index
        candidates = [
            self.project_root / "data" / "cleaned" / "content_goren_2025" / "key2txt.tsv",
            self.project_root / "data" / "cleaned" / "content" / "key2txt.tsv",
            self.project_root / "data" / "cleaned" / "index" / "key2txt.tsv",
        ]
        merged: dict[str, Path] = {}
        for p in candidates:
            if not p.exists():
                continue
            m = self._read_key2txt_with_header(p)
            if not m:
                m = self._read_key2txt_no_header(p)
            for k, v in m.items():
                if v.exists():
                    merged[k] = v
        self.key2txt_map = merged

    def _build_table_index(self) -> None:
        table_roots = [
            self.project_root / "data" / "cleaned" / "content_goren_2025" / "tables",
            self.project_root / "data" / "cleaned" / "content" / "tables",
        ]
        idx: dict[str, list[Path]] = {}
        meta: list[dict[str, Any]] = []
        for root in table_roots:
            if not root.exists():
                continue
            for p in root.rglob("*.csv"):
                idx.setdefault(p.name.lower(), []).append(p.resolve())
                cols: list[str] = []
                caption = ""
                try:
                    hdf = pd.read_csv(p, dtype=str)
                    cols = [str(c) for c in hdf.columns]
                    if not hdf.empty:
                        first_vals = [str(x) for x in hdf.iloc[0].tolist() if str(x).strip()]
                        caption = " | ".join(first_vals[:3])
                except Exception:
                    cols = []
                    caption = ""
                cols_norm = [re.sub(r"\s+", " ", c.lower()).strip() for c in cols]
                has_target_cols = any(
                    re.search(r"\bee\b|encapsulation|particle size|\bsize\b|\bpdi\b|drug|polymer|surfactant|x\d|coded|level|low|high", c)
                    for c in cols_norm
                )
                is_doe_cols = any(re.match(r"^x\d+$", c.replace(" ", "")) for c in cols_norm) or any(
                    re.search(r"coded|level|low|medium|high", c) for c in cols_norm
                )
                meta.append(
                    {
                        "path": p.resolve(),
                        "name": p.name.lower(),
                        "columns": cols,
                        "columns_norm": cols_norm,
                        "caption": caption,
                        "has_target_cols": has_target_cols,
                        "is_doe_cols": is_doe_cols,
                    }
                )
        self.table_file_index = idx
        self.table_meta = meta

    def _load_zotero_table_metadata(self) -> None:
        candidates = [
            self.project_root / "data" / "raw" / "zotero" / "zotero_collection__goren_2025.jsonl",
            self.project_root / "data" / "raw" / "zotero" / "zotero_llm_relevant.jsonl",
            self.project_root / "data" / "raw" / "zotero" / "zotero_selected_items.jsonl",
        ]
        out: dict[str, list[Path]] = {}
        for p in candidates:
            if not p.exists():
                continue
            try:
                with p.open("r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        key = str(obj.get("zotero_key", "")).strip()
                        if not key:
                            continue
                        paths_obj = obj.get("paths", {}) if isinstance(obj.get("paths", {}), dict) else {}
                        tsv = paths_obj.get("tables_csv", []) if isinstance(paths_obj, dict) else []
                        if not isinstance(tsv, list):
                            continue
                        for rel in tsv:
                            rp = str(rel).strip()
                            if not rp:
                                continue
                            full = (self.project_root / rp).resolve() if not Path(rp).is_absolute() else Path(rp).resolve()
                            if full.exists():
                                out.setdefault(key, []).append(full)
            except Exception:
                continue
        self.table_paths_by_key = {k: sorted(set(v), key=lambda x: str(x)) for k, v in out.items()}

    def _norm_doi_token(self, doi: str) -> str:
        s = str(doi or "").lower().strip()
        s = re.sub(r"^doi\s*:\s*", "", s)
        s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
        return re.sub(r"[^a-z0-9]+", "", s)

    def _path_is_for_key(self, p: Path, zotero_key: str) -> bool:
        s = str(p.resolve()).replace("\\", "/").lower()
        k = str(zotero_key or "").strip().lower()
        return bool(k) and f"/tables/{k}/" in s

    def _log_drop_nonlocal(self, zotero_key: str, p: Path, source: str) -> None:
        print(
            f"[resolver_drop_nonlocal] key={zotero_key} source={source} path={p}"
        )

    def _read_key_tables_manifest(self, key_dir: Path, zotero_key: str) -> list[Path]:
        manifest = key_dir / "tables_manifest.json"
        if not manifest.exists():
            return []
        out: list[Path] = []
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return []

        def _to_path(v: Any) -> Path | None:
            s = str(v or "").strip()
            if not s:
                return None
            p = Path(s)
            if not p.is_absolute():
                p = (self.project_root / p).resolve()
            else:
                p = p.resolve()
            return p

        if isinstance(payload, dict):
            for x in payload.get("selected_table_files", []) if isinstance(payload.get("selected_table_files", []), list) else []:
                p = _to_path(x)
                if p is not None:
                    out.append(p)
            if not out:
                for rec in payload.get("tables", []) if isinstance(payload.get("tables", []), list) else []:
                    if not isinstance(rec, dict):
                        continue
                    p = _to_path(rec.get("csv_path", "") or rec.get("filename", ""))
                    if p is not None:
                        out.append(p)
        elif isinstance(payload, list):
            for rec in payload:
                if isinstance(rec, dict):
                    p = _to_path(rec.get("csv_path", "") or rec.get("filename", ""))
                    if p is not None:
                        out.append(p)
        # Keep only existing CSVs under this key dir.
        key_dir_resolved = key_dir.resolve()
        filtered: list[Path] = []
        for p in out:
            try:
                p.resolve().relative_to(key_dir_resolved)
            except Exception:
                continue
            if p.exists() and p.suffix.lower() == ".csv":
                filtered.append(p.resolve())
        return sorted(set(filtered), key=lambda x: str(x))

    def _paper_local_tables(self, zotero_key: str, doi: str = "", title: str = "") -> list[dict[str, Any]]:
        k = str(zotero_key or "").strip()
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        if not k:
            return out

        key_dir = (
            self.project_root / "data" / "cleaned" / "content_goren_2025" / "tables" / k
        ).resolve()
        if not key_dir.exists():
            return out

        # (1) Prefer explicit per-key tables manifest when present.
        manifest_paths = self._read_key_tables_manifest(key_dir=key_dir, zotero_key=k)
        for p in manifest_paths:
            if not self._path_is_for_key(p, k):
                self._log_drop_nonlocal(k, p, "per_key_manifest")
                continue
            sp = str(p.resolve())
            if sp in seen:
                continue
            seen.add(sp)
            out.append({"path": p.resolve(), "source": "per_key_manifest"})

        # (2) Fallback: only scan within this key directory.
        if not out:
            for p in key_dir.rglob("*.csv"):
                rp = p.resolve()
                if not self._path_is_for_key(rp, k):
                    self._log_drop_nonlocal(k, rp, "per_key_glob")
                    continue
                sp = str(rp)
                if sp in seen:
                    continue
                seen.add(sp)
                out.append({"path": rp, "source": "per_key_glob"})

        # (3) Optional metadata paths, but hard-restricted to this key folder.
        for p in self.table_paths_by_key.get(k, []):
            rp = p.resolve()
            if not self._path_is_for_key(rp, k):
                self._log_drop_nonlocal(k, rp, "metadata_tables_csv")
                continue
            sp = str(rp)
            if sp in seen:
                continue
            seen.add(sp)
            out.append({"path": rp, "source": "metadata_tables_csv"})

        # Unit-like self-check guard: never return cross-key table paths.
        checked: list[dict[str, Any]] = []
        for rec in out:
            rp = Path(rec["path"]).resolve()
            if not self._path_is_for_key(rp, k):
                self._log_drop_nonlocal(k, rp, str(rec.get("source", "unknown")))
                continue
            checked.append(rec)
        return checked

    def ownership_check_passed(
        self,
        chosen_table_csv: Path | None,
        paper_local_table_files: set[str],
        zotero_key: str,
        doi: str,
    ) -> tuple[bool, str]:
        if chosen_table_csv is None:
            return False, "no_chosen_table"
        chosen = str(chosen_table_csv.resolve())
        if chosen in paper_local_table_files:
            return True, "chosen_table_in_paper_local_registry"
        base = chosen_table_csv.name.lower()
        key_hit = bool(str(zotero_key or "").strip().lower() in base) if zotero_key else False
        doi_tok = self._norm_doi_token(doi)
        doi_hit = bool(doi_tok and doi_tok in re.sub(r"[^a-z0-9]+", "", base))
        return (
            False,
            f"chosen_table_not_in_registry;diag_key_hit={key_hit};diag_doi_hit={doi_hit}",
        )

    def resolve_text_path(self, zotero_key: str) -> Path | None:
        k = str(zotero_key).strip()
        if not k:
            return None
        if k in self.key2txt_map and self.key2txt_map[k].exists():
            return self.key2txt_map[k]

        fallback_candidates = [
            self.project_root / "data" / "cleaned" / "content_goren_2025" / "text" / f"{k}.pdf.txt",
            self.project_root / "data" / "cleaned" / "content_goren_2025" / "text" / f"{k}.html.txt",
            self.project_root / "data" / "cleaned" / "content" / "text" / f"{k}.pdf.txt",
            self.project_root / "data" / "cleaned" / "content" / "text" / f"{k}.html.txt",
        ]
        for p in fallback_candidates:
            if p.exists():
                return p.resolve()
        return None

    def _load_text(self, path: Path | None) -> str:
        if path is None:
            return ""
        key = str(path.resolve())
        if key in self.text_cache:
            return self.text_cache[key]
        try:
            txt = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            txt = ""
        self.text_cache[key] = txt
        return txt

    def resolve_text_evidence(
        self,
        zotero_key: str,
        evidence_span_start: Any,
        evidence_span_end: Any,
        evidence_section: Any,
        evidence_pointer_raw: str,
        max_span_chars: int,
        fallback_hint_text: str = "",
    ) -> TextEvidence:
        text_path = self.resolve_text_path(zotero_key)
        full = self._load_text(text_path)
        sec = str(evidence_section or "")
        start_s = str(evidence_span_start or "").strip()
        end_s = str(evidence_span_end or "").strip()
        block_id = sec
        span_id = f"{start_s}-{end_s}" if start_s and end_s else ""

        if full and start_s.isdigit() and end_s.isdigit():
            s = max(0, int(start_s))
            e = max(s, int(end_s))
            if s < len(full):
                e = min(len(full), e)
                core = full[s:e]
                # Expand if too short or mostly whitespace.
                if len(core.strip()) < 20:
                    s2 = max(0, s - 200)
                    e2 = min(len(full), e + 200)
                    core = full[s2:e2]
                before = full[max(0, s - 150):s]
                after = full[e:min(len(full), e + 150)]
                return TextEvidence(
                    evidence_source_type="fulltext",
                    evidence_pointer_raw=evidence_pointer_raw,
                    evidence_text=short_text(core, max_span_chars),
                    evidence_context_before=short_text(before, 150),
                    evidence_context_after=short_text(after, 150),
                    evidence_block_id=block_id,
                    evidence_span_id=span_id,
                    evidence_span_start=start_s,
                    evidence_span_end=end_s,
                    evidence_section=sec,
                )

        # Best-effort fallback to hint text (not pointer-only).
        fallback = short_text(fallback_hint_text, max_span_chars)
        return TextEvidence(
            evidence_source_type="unknown",
            evidence_pointer_raw=evidence_pointer_raw,
            evidence_text=fallback,
            evidence_context_before="",
            evidence_context_after="",
            evidence_block_id=block_id,
            evidence_span_id=span_id,
            evidence_span_start=start_s,
            evidence_span_end=end_s,
            evidence_section=sec,
        )

    def _extract_table_csv_name(self, pointer_raw: str) -> str:
        p = str(pointer_raw or "")
        m = re.search(r"([A-Za-z0-9._-]+\.csv)", p, flags=re.IGNORECASE)
        return m.group(1) if m else ""

    def _table_rows_for_match(
        self,
        df: pd.DataFrame,
        target_values: dict[str, str],
    ) -> tuple[int, str]:
        # Return best row idx and row text by matching numeric targets.
        if df.empty:
            return -1, ""
        best_idx = -1
        best_score = -1.0
        target_nums: list[float] = []
        for _, v in target_values.items():
            s = str(v or "").strip()
            if not s:
                continue
            m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
            if not m:
                continue
            try:
                target_nums.append(float(m.group(0)))
            except Exception:
                pass
        for i in range(len(df)):
            row = df.iloc[i]
            row_text = " | ".join([f"{c}:{row[c]}" for c in df.columns if str(row[c]).strip()])
            score = 0.0
            if target_nums:
                for c in df.columns:
                    try:
                        rv = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
                    except Exception:
                        rv = None
                    if rv is None or pd.isna(rv):
                        continue
                    for tv in target_nums:
                        if abs(float(rv) - float(tv)) < 1e-6:
                            score += 2.0
                        elif abs(float(rv) - float(tv)) <= max(0.05 * abs(tv), 0.5):
                            score += 1.0
            # Prefer row with richer content if tie.
            score += min(0.01 * len(row_text), 1.0)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            return -1, ""
        rr = df.iloc[best_idx]
        text = " | ".join([f"{c}:{rr[c]}" for c in df.columns if str(rr[c]).strip()])
        return best_idx, text

    def _extract_doe_signature_from_row(self, row: pd.Series, cols: list[str]) -> str:
        parts: list[str] = []
        for c in cols:
            cn = c.lower().replace(" ", "")
            if re.match(r"^x\d+$", cn):
                val = str(row.get(c, "")).strip()
                if val:
                    parts.append(f"{cn.upper()}={val}")
        if parts:
            return "|".join(parts)
        # Fallback for coded levels table columns.
        keys = []
        for c in cols:
            cl = c.lower()
            if re.search(r"coded|level|low|medium|high", cl):
                v = str(row.get(c, "")).strip()
                if v:
                    keys.append(f"{c}={v}")
        return "|".join(keys[:6])

    def detect_doe_keywords(self, text_blob: str) -> bool:
        t = str(text_blob or "").lower()
        return bool(re.search(r"factorial|box[- ]?behnken|coded|levels?", t))

    def resolve_table_evidence(
        self,
        zotero_key: str,
        doi: str,
        title: str,
        pointer_raw: str,
        row_index: Any = "",
        col_name: str = "",
        target_values: dict[str, str] | None = None,
        field_hint: str = "",
        target_field: str = "",
        notes_hint: str = "",
        max_table_row_chars: int = 800,
    ) -> TableEvidence:
        target_values = target_values or {}
        csv_name = self._extract_table_csv_name(pointer_raw)
        direct_paths = self.table_file_index.get(csv_name.lower(), []) if csv_name else []

        # Build authoritative paper-local registry only from paper-local signals.
        local = self._paper_local_tables(zotero_key=zotero_key, doi=doi, title=title)
        paper_local_registry = set([str(x["path"].resolve()) for x in local if str(x.get("path", "")).strip()])
        local_map = {str(x["path"]): x for x in local}

        candidates: list[dict[str, Any]] = []
        for p in direct_paths:
            sp = str(p)
            if sp in local_map:
                candidates.append(local_map[sp])
        for x in local:
            if str(x["path"]) not in {str(c["path"]) for c in candidates}:
                candidates.append(x)

        if not candidates:
            # Fulltext proxy fallback for DoE rows when the paper has zero extracted tables.
            notes = str(notes_hint or "")
            vals = {k: str(v).strip() for k, v in target_values.items() if str(v).strip()}
            if vals:
                kv = " | ".join([f"{k}={v}" for k, v in vals.items()])
                base = notes if notes else ""
                combined = f"{base} | extracted_row_values: {kv}".strip(" |")
                is_doe_like = bool(re.search(r"table\s*\d|factorial|box[- ]?behnken|x\d|coded|level", notes, flags=re.IGNORECASE))
                return TableEvidence(
                    table_csv_path="",
                    table_filename=f"fulltext_table_proxy__{zotero_key}",
                    rejected_table_filename=csv_name,
                    table_title_or_caption=(
                        "Table evidence proxy from paper-local fulltext note"
                        if is_doe_like
                        else "Table evidence proxy from paper-local extracted row values"
                    ),
                    table_match_score=0.3 if is_doe_like else 0.2,
                    table_row_text=short_text(combined, max_table_row_chars),
                    table_cell_text="",
                    doe_signature="",
                    top5_candidates=[],
                    top5_scores=[],
                    match_reason="paper_local_csv_missing_fulltext_proxy_used",
                    paper_local_candidate_count=int(len(paper_local_registry)),
                    ownership_check_passed=False,
                    ownership_check_reason="no_paper_local_candidates_proxy_used",
                    chosen_table_rejected=bool(csv_name),
                    table_evidence_missing_reason="",
                )
            return TableEvidence(
                table_csv_path="",
                table_filename="",
                rejected_table_filename=csv_name,
                table_title_or_caption="",
                table_match_score=0.0,
                table_row_text="",
                table_cell_text="",
                doe_signature="",
                top5_candidates=[],
                top5_scores=[],
                match_reason="",
                paper_local_candidate_count=int(len(paper_local_registry)),
                ownership_check_passed=False,
                ownership_check_reason="no_paper_local_candidates",
                chosen_table_rejected=bool(csv_name),
                table_evidence_missing_reason="no_paper_local_candidate_table_csv",
            )

        best: dict[str, Any] = {
            "path": None,
            "row_idx": -1,
            "row_text": "",
            "cell_text": "",
            "caption": "",
            "score": -1.0,
            "doe_signature": "",
            "reason": "",
        }
        scored: list[tuple[str, float]] = []
        for cand in candidates:
            p = cand["path"]
            try:
                df = pd.read_csv(p, dtype=str).fillna("")
            except Exception:
                continue
            cols = [str(c) for c in df.columns]
            cols_blob = " ".join([c.lower() for c in cols])
            caption = ""
            meta_hit = [m for m in self.table_meta if str(m["path"]) == str(p)]
            if meta_hit:
                caption = str(meta_hit[0].get("caption", ""))
            hint_blob = f"{cols_blob} {caption.lower()} {p.name.lower()}"
            # Row selection priority: explicit row index -> value match -> first row.
            ridx = -1
            row_text = ""
            if str(row_index).strip().isdigit():
                idx = int(str(row_index).strip())
                if 0 <= idx < len(df):
                    rr = df.iloc[idx]
                    ridx = idx
                    row_text = " | ".join([f"{c}:{rr[c]}" for c in cols if str(rr[c]).strip()])
            if ridx < 0:
                ridx, row_text = self._table_rows_for_match(df, target_values=target_values)
            if ridx < 0 and len(df) > 0:
                ridx = 0
                rr = df.iloc[ridx]
                row_text = " | ".join([f"{c}:{rr[c]}" for c in cols if str(rr[c]).strip()])
            if ridx < 0:
                continue
            rr = df.iloc[ridx]
            cell_text = ""
            if col_name and col_name in df.columns:
                cell_text = str(rr.get(col_name, ""))
            # Score by direct pointer + value match richness.
            score = 0.0
            reason_parts: list[str] = []
            if cand["source"] == "metadata_tables_csv":
                score += 4.0
                reason_parts.append("paper_metadata_table")
            elif cand["source"] == "filename_key_or_docid":
                score += 3.0
                reason_parts.append("filename_key_or_docid")
            elif "title_token_match" in cand["source"]:
                score += 2.0
                reason_parts.append(cand["source"])
            if csv_name and p.name.lower() == csv_name.lower():
                score += 2.0
                reason_parts.append("pointer_filename_match")
            if re.search(r"factorial|design|layout|x1|x2|x3|\bee\b|\bps\b|particle size|\bpdi\b", hint_blob):
                score += 2.0
                reason_parts.append("header_or_caption_target_hit")
            score += min(len(row_text) / 200.0, 2.0)
            value_hits = 0
            row_num_blob = " ".join([str(v) for v in rr.tolist()])
            for _, tv in target_values.items():
                sv = str(tv).strip()
                if not sv:
                    continue
                m = re.search(r"[-+]?\d+(?:\.\d+)?", sv)
                if not m:
                    continue
                if m.group(0) in row_num_blob:
                    value_hits += 1
            if value_hits:
                score += min(2.0, 0.8 * value_hits)
                reason_parts.append(f"value_match_{value_hits}")
            if self._extract_doe_signature_from_row(rr, cols):
                score += 1.0
                reason_parts.append("doe_row_signature")
            scored.append((p.name, round(float(score), 4)))
            if score > best["score"]:
                best["path"] = p
                best["row_idx"] = ridx
                best["row_text"] = row_text
                best["cell_text"] = cell_text
                best["caption"] = caption
                best["score"] = score
                best["doe_signature"] = self._extract_doe_signature_from_row(rr, cols)
                best["reason"] = "|".join(reason_parts)

        if best["path"] is None:
            return TableEvidence(
                table_csv_path="",
                table_filename="",
                rejected_table_filename=csv_name,
                table_title_or_caption="",
                table_match_score=0.0,
                table_row_text="",
                table_cell_text="",
                doe_signature="",
                top5_candidates=[],
                top5_scores=[],
                match_reason="",
                paper_local_candidate_count=int(len(paper_local_registry)),
                ownership_check_passed=False,
                ownership_check_reason="table_search_failed_within_paper_local_candidates",
                chosen_table_rejected=bool(csv_name),
                table_evidence_missing_reason="table_search_failed",
            )
        scored = sorted(scored, key=lambda x: x[1], reverse=True)[:5]

        chosen_path = best["path"]
        ownership_ok, ownership_reason = self.ownership_check_passed(
            chosen_table_csv=chosen_path,
            paper_local_table_files=paper_local_registry,
            zotero_key=zotero_key,
            doi=doi,
        )
        if not ownership_ok:
            notes = str(notes_hint or "")
            vals = {k: str(v).strip() for k, v in target_values.items() if str(v).strip()}
            if vals:
                kv = " | ".join([f"{k}={v}" for k, v in vals.items()])
                combined = f"{notes} | extracted_row_values: {kv}".strip(" |")
                return TableEvidence(
                    table_csv_path="",
                    table_filename=f"fulltext_table_proxy__{zotero_key}",
                    rejected_table_filename=str(Path(str(chosen_path)).name),
                    table_title_or_caption="Rejected non-owned table; fallback proxy from paper-local values",
                    table_match_score=0.0,
                    table_row_text=short_text(combined, max_table_row_chars),
                    table_cell_text="",
                    doe_signature="",
                    top5_candidates=[x[0] for x in scored],
                    top5_scores=[x[1] for x in scored],
                    match_reason="ownership_check_failed_downgraded_to_proxy",
                    paper_local_candidate_count=int(len(paper_local_registry)),
                    ownership_check_passed=False,
                    ownership_check_reason=ownership_reason,
                    chosen_table_rejected=True,
                    table_evidence_missing_reason="",
                )
            return TableEvidence(
                table_csv_path="",
                table_filename="",
                rejected_table_filename=str(Path(str(chosen_path)).name),
                table_title_or_caption="",
                table_match_score=0.0,
                table_row_text="",
                table_cell_text="",
                doe_signature="",
                top5_candidates=[x[0] for x in scored],
                top5_scores=[x[1] for x in scored],
                match_reason="ownership_check_failed_downgraded_to_none",
                paper_local_candidate_count=int(len(paper_local_registry)),
                ownership_check_passed=False,
                ownership_check_reason=ownership_reason,
                chosen_table_rejected=True,
                table_evidence_missing_reason="ownership_check_failed",
            )

        return TableEvidence(
            table_csv_path=str(chosen_path),
            table_filename=str(Path(str(chosen_path)).name),
            rejected_table_filename="",
            table_title_or_caption=short_text(best["caption"], 300),
            table_match_score=round(float(best["score"]), 4),
            table_row_text=short_text(best["row_text"], max_table_row_chars),
            table_cell_text=short_text(best["cell_text"], max_table_row_chars),
            doe_signature=short_text(best["doe_signature"], 200),
            top5_candidates=[x[0] for x in scored],
            top5_scores=[x[1] for x in scored],
            match_reason=str(best["reason"]),
            paper_local_candidate_count=int(len(paper_local_registry)),
            ownership_check_passed=True,
            ownership_check_reason=ownership_reason,
            chosen_table_rejected=False,
            table_evidence_missing_reason="",
        )
