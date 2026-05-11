#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"


FIELD_SPECS: dict[str, dict[str, Any]] = {
    "drug_feed_amount_text": {
        "unit_tokens": ["mg", "g", "ug", "mcg", "ng", "mg/ml", "g/ml", "%w/w"],
        "entity_type": "drug",
    },
    "plga_mass_mg": {
        "unit_tokens": ["mg", "g", "ug", "mcg", "ng", "mg/ml", "g/ml"],
        "entity_type": "polymer",
    },
    "pva_conc_percent": {
        "unit_tokens": ["%", "percent", "wt%", "w/v", "% w/v", "%w/v"],
        "entity_type": "surfactant",
    },
    "encapsulation_efficiency_percent": {
        "unit_tokens": ["%", "percent"],
        "entity_type": "",
    },
    "loading_content_percent": {
        "unit_tokens": ["%", "percent"],
        "entity_type": "",
    },
    "size_nm": {
        "unit_tokens": ["nm", "nanometer", "nanometers"],
        "entity_type": "",
    },
    "polymer_mw_kDa": {
        "unit_tokens": ["kda", "da", "mw"],
        "entity_type": "polymer",
    },
    "la_ga_ratio": {
        "unit_tokens": [":"],
        "entity_type": "polymer",
    },
}

CORE_FIELDS = [
    "encapsulation_efficiency_percent",
    "loading_content_percent",
    "drug_feed_amount_text",
    "plga_mass_mg",
    "la_ga_ratio",
    "polymer_mw_kDa",
    "size_nm",
    "pva_conc_percent",
]

FINGERPRINT_FIELDS = [
    "emul_type",
    "emul_method",
    "la_ga_ratio",
    "polymer_mw_kDa",
    "plga_mass_mg",
    "pva_conc_percent",
    "organic_solvent",
    "drug_name",
    "drug_feed_amount_text",
]

FIELD_ALIASES = {
    "polymer_mw_kDa": ["polymer_mw_kDa", "plga_mw_kDa"],
}

DEPRIORITIZED_SECTION_HINTS = {
    "abstract",
    "introduction",
    "background",
}

CONSTANT_PARAMETER_PATTERNS = [
    re.compile(r"all other parameters (were )?kept constant", re.IGNORECASE),
    re.compile(r"other parameters (were )?kept constant", re.IGNORECASE),
    re.compile(r"remaining parameters (were )?kept constant", re.IGNORECASE),
    re.compile(r"while .* was varied", re.IGNORECASE),
    re.compile(r"except .* all other", re.IGNORECASE),
]

EE_HEADER_HINT_PATTERNS = [
    re.compile(r"encapsulation\s+efficiency", re.IGNORECASE),
    re.compile(r"entrapment\s+efficiency", re.IGNORECASE),
    re.compile(r"\bdee\b", re.IGNORECASE),
    re.compile(r"\bee\b", re.IGNORECASE),
]

LOADING_HEADER_HINT_PATTERNS = [
    re.compile(r"drug\s+loading", re.IGNORECASE),
    re.compile(r"loading\s+content", re.IGNORECASE),
    re.compile(r"\blc\b", re.IGNORECASE),
    re.compile(r"\bdl\b", re.IGNORECASE),
    re.compile(r"\bloading\b", re.IGNORECASE),
]

RATIO_HEADER_HINT_PATTERNS = [
    re.compile(r"drug\s*/\s*polymer", re.IGNORECASE),
    re.compile(r"\bd\s*/\s*p\b", re.IGNORECASE),
    re.compile(r"\bratio\b", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run evidence-token gating QC on extracted numeric fields and flag unsupported values."
    )
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--input-tsv", default="", help="Default: data/results/<run_id>/weak_labels__gemini.tsv")
    p.add_argument(
        "--out-dir",
        default="",
        help="Default: data/results/<run_id>/benchmark_goren_2025/derivation_v1",
    )
    p.add_argument("--sample-manifest", default="data/cleaned/samples/sample_goren18.tsv")
    p.add_argument("--key2txt", default="data/cleaned/content_goren_2025/key2txt.tsv")
    p.add_argument(
        "--realign-log",
        default="",
        help="Default: data/results/<run_id>/evidence_realign_log.tsv",
    )
    p.add_argument("--boilerplate-repeat-threshold", type=int, default=3)
    p.add_argument("--boilerplate-min-line-len", type=int, default=20)
    p.add_argument("--numeric-rel-tol", type=float, default=1e-3)
    p.add_argument("--numeric-abs-tol", type=float, default=1e-6)
    p.add_argument("--unit-min-matches", type=int, default=1)
    p.add_argument("--entity-min-matches", type=int, default=1)
    p.add_argument("--require-unit-token", action="store_true", default=True)
    p.add_argument("--no-require-unit-token", dest="require_unit_token", action="store_false")
    p.add_argument("--require-entity-for-mass", action="store_true", default=True)
    p.add_argument("--no-require-entity-for-mass", dest="require_entity_for_mass", action="store_false")
    p.add_argument(
        "--ee-structured-mode",
        choices=["legacy", "table_block"],
        default="table_block",
        help="legacy: header hint in EE span only; table_block: table-aware context around span/table chunk.",
    )
    p.add_argument(
        "--ee-window-chars",
        type=int,
        default=1800,
        help="Context window size around span start for EE table-context fallback.",
    )
    p.add_argument(
        "--audit-pack-xlsx",
        default="",
        help="Optional audit workbook path. If set, QC can use table_cell_text/table_row_text for EE/size matching.",
    )
    return p.parse_args()


def normalize_text(v: Any) -> str:
    s = str(v or "").lower()
    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    return s


def tokenize_name(v: str) -> list[str]:
    toks = re.findall(r"[a-zA-Z]{3,}", str(v or "").lower())
    stop = {"the", "and", "for", "with", "from", "into", "drug"}
    out = [t for t in toks if t not in stop]
    return sorted(set(out))


def find_numeric_tokens(v: str) -> list[str]:
    raw = re.findall(r"[+-]?\d+(?:[.,]\d+)?", str(v or ""))
    cleaned = []
    for x in raw:
        x = x.replace(",", ".")
        try:
            num = float(x)
        except ValueError:
            continue
        if abs(num) < 1e-12:
            cleaned.append("0")
        else:
            cleaned.append(f"{num:.12f}".rstrip("0").rstrip("."))
    return cleaned


def build_required_unit_tokens(field_name: str, value_text: str) -> list[str]:
    spec_units = FIELD_SPECS.get(field_name, {}).get("unit_tokens", [])
    v = normalize_text(value_text)
    detected = []
    for u in spec_units:
        if u in v:
            detected.append(u)
    if detected:
        return sorted(set(detected))
    return sorted(set(spec_units))


def build_required_entity_tokens(field_name: str, row: pd.Series) -> list[str]:
    et = FIELD_SPECS.get(field_name, {}).get("entity_type", "")
    if et == "polymer":
        return ["plga", "poly(lactic-co-glycolic acid)", "poly(lactide-co-glycolide)"]
    if et == "surfactant":
        return ["pva", "polyvinyl alcohol", "poloxamer", "surfactant"]
    if et == "drug":
        return tokenize_name(str(row.get("drug_name", "")).strip())
    return []


def canonicalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for canonical_name, aliases in FIELD_ALIASES.items():
        canonical_value = out[canonical_name] if canonical_name in out.columns else pd.Series([""] * len(out), index=out.index)
        for alias in aliases:
            if alias == canonical_name or alias not in out.columns:
                continue
            canonical_value = canonical_value.where(canonical_value.astype(str).str.strip().ne(""), out[alias])
        out[canonical_name] = canonical_value
    return out


def contains_any(text: str, tokens: list[str]) -> int:
    if not tokens:
        return 0
    c = 0
    for t in tokens:
        tt = normalize_text(t).strip()
        if not tt:
            continue
        if tt in text:
            c += 1
    return c


def short_text(v: str, n: int = 220) -> str:
    s = str(v or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def normalize_value_token(v: Any) -> str:
    s = normalize_text(v).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def is_explicit_formulation_id(fid: str) -> bool:
    f = str(fid or "").strip()
    if not f:
        return False
    # Numeric-only IDs are treated as non-explicit labels for this use case.
    if re.fullmatch(r"\d+", f):
        return False
    # Typical explicit formulation labels: F1/F-1/NPR2/etc.
    return bool(re.search(r"[A-Za-z]", f))


def parse_figure_group_label(text: str) -> str:
    t = str(text or "")
    m = re.search(r"\b(?:fig(?:ure)?\.?\s*\d+[a-z]?|group\s*[A-Z0-9]+)\b", t, flags=re.IGNORECASE)
    if m:
        return normalize_value_token(m.group(0))
    return ""


def parse_enumerated_condition_label(text: str) -> str:
    t = str(text or "")
    m = re.search(
        r"\b(?:run|condition|entry|sample|set)\s*[-:#]?\s*(\d+[a-z]?)\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        return normalize_value_token(m.group(0))
    return ""


def build_condition_instance_key(row: pd.Series) -> tuple[str, str]:
    key = str(row.get("key", "")).strip()
    fid = str(row.get("formulation_id", "")).strip()
    notes = str(row.get("notes", "")).strip()
    evidence = str(row.get("evidence_span_text", "")).strip()

    if is_explicit_formulation_id(fid):
        return f"{key}::id::{fid}", "explicit_formulation_id"

    # 1) Prefer table-row condition fingerprint from row conditions.
    condition_pairs: list[str] = []
    for f in FINGERPRINT_FIELDS:
        if f in row.index:
            v = str(row.get(f, "")).strip()
            if v:
                condition_pairs.append(f"{f}={normalize_value_token(v)}")
    if len(condition_pairs) >= 2:
        fp_text = "|".join(sorted(condition_pairs))
        fp_hash = hashlib.sha1(fp_text.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"{key}::cond::{fp_hash}", "table_row_conditions"

    # 2) Figure/group labels.
    fig_label = parse_figure_group_label(f"{notes}\n{evidence}\n{fid}")
    if fig_label:
        h = hashlib.sha1(fig_label.encode("utf-8", errors="ignore")).hexdigest()[:10]
        return f"{key}::fig::{h}", "figure_group_label"

    # 3) Enumerated text conditions.
    enum_label = parse_enumerated_condition_label(f"{notes}\n{evidence}\n{fid}")
    if enum_label:
        h = hashlib.sha1(enum_label.encode("utf-8", errors="ignore")).hexdigest()[:10]
        return f"{key}::enum::{h}", "enumerated_text_condition"

    # Stable fallback.
    row_idx = row.name if row.name is not None else "na"
    return f"{key}::row::{row_idx}", "fallback_row_index"


def detect_constant_claim(text: str) -> bool:
    t = str(text or "")
    return any(p.search(t) is not None for p in CONSTANT_PARAMETER_PATTERNS)


def has_ee_header_hint(text: str) -> bool:
    t = str(text or "")
    return any(p.search(t) is not None for p in EE_HEADER_HINT_PATTERNS)


def has_loading_header_hint(text: str) -> bool:
    t = str(text or "")
    return any(p.search(t) is not None for p in LOADING_HEADER_HINT_PATTERNS)


def has_ratio_header_hint(text: str) -> bool:
    t = str(text or "")
    return any(p.search(t) is not None for p in RATIO_HEADER_HINT_PATTERNS)


def parse_first_number(value_text: str) -> float | None:
    toks = find_numeric_tokens(value_text)
    if not toks:
        return None
    try:
        return float(toks[0])
    except ValueError:
        return None


def get_char_window(text: str, center: str, half_window: int) -> str:
    src = str(text or "")
    if not src:
        return ""
    if str(center).isdigit():
        c = int(center)
        lo = max(0, c - int(half_window))
        hi = min(len(src), c + int(half_window))
        return src[lo:hi]
    return src[: min(len(src), int(2 * half_window))]


def is_table_like_chunk(text: str) -> bool:
    lines = [ln for ln in str(text or "").splitlines() if ln.strip()]
    if not lines:
        return False
    tableish = 0
    for ln in lines:
        nnums = len(re.findall(r"[+-]?\d+(?:[.,]\d+)?", ln))
        if ("\t" in ln) or ("|" in ln) or (nnums >= 2):
            tableish += 1
    return tableish >= 2


def locate_line_index_by_char(text: str, pos: int) -> int:
    if pos < 0:
        return 0
    cur = 0
    for i, ln in enumerate(str(text).splitlines(True)):
        nxt = cur + len(ln)
        if cur <= pos < nxt:
            return i
        cur = nxt
    return max(0, len(str(text).splitlines()) - 1)


def get_line_window(text: str, center: str, half_lines: int = 20) -> str:
    lines = str(text or "").splitlines()
    if not lines:
        return ""
    if str(center).isdigit():
        idx = locate_line_index_by_char(str(text), int(center))
    else:
        idx = 0
    lo = max(0, idx - int(half_lines))
    hi = min(len(lines), idx + int(half_lines) + 1)
    return "\n".join(lines[lo:hi])


def compute_key_level_constant_claims(all_rows: pd.DataFrame) -> dict[str, bool]:
    claims: dict[str, bool] = {}
    for key, g in all_rows.groupby("key", sort=False):
        blob = "\n".join(
            [
                str(x)
                for x in (
                    list(g.get("notes", pd.Series(dtype=str)).astype(str).tolist())
                    + list(g.get("evidence_span_text", pd.Series(dtype=str)).astype(str).tolist())
                )
            ]
        )
        claims[str(key)] = detect_constant_claim(blob)
    return claims


def choose_doc_polymer_donors(
    all_rows: pd.DataFrame,
    checks: pd.DataFrame,
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], bool]]:
    """
    Returns:
    - donors[(key, field)] = {"value": normalized_value, "group_key": donor_group_key}
    - conflicts[(key, field)] = True when multiple distinct candidates exist.
    """
    donor_fields = ["la_ga_ratio", "polymer_mw_kDa"]
    donors: dict[tuple[str, str], dict[str, str]] = {}
    conflicts: dict[tuple[str, str], bool] = {}

    if all_rows.empty:
        return donors, conflicts

    support_map: dict[tuple[str, str, str], int] = {}
    if not checks.empty:
        c = checks.copy()
        c["evidence_supported"] = pd.to_numeric(c.get("evidence_supported", 0), errors="coerce").fillna(0).astype(int)
        for _, r in c.iterrows():
            key = str(r.get("key", "")).strip()
            gk = str(r.get("group_key", "")).strip()
            field = str(r.get("field_name", "")).strip()
            if field in donor_fields and key and gk:
                support_map[(key, gk, field)] = int(r.get("evidence_supported", 0))

    row_copy = all_rows.copy()
    row_copy["group_key"] = row_copy["key"].astype(str).str.strip() + "::" + row_copy["formulation_id"].astype(str).str.strip()

    for key, g in row_copy.groupby("key", sort=False):
        key_s = str(key).strip()
        for field in donor_fields:
            if field not in g.columns:
                conflicts[(key_s, field)] = True
                continue
            cand_rows = []
            for _, rr in g.iterrows():
                raw_v = str(rr.get(field, "")).strip()
                if not raw_v:
                    continue
                norm_v = normalize_value_token(raw_v)
                gk = str(rr.get("group_key", "")).strip()
                sup = int(support_map.get((key_s, gk, field), 0))
                cand_rows.append((norm_v, gk, sup))

            if not cand_rows:
                conflicts[(key_s, field)] = False
                continue

            supported = [x for x in cand_rows if x[2] == 1]
            candidate_pool = supported if supported else cand_rows
            distinct = sorted(set(v for v, _, _ in candidate_pool if v))
            if len(distinct) != 1:
                conflicts[(key_s, field)] = True
                continue

            chosen_v = distinct[0]
            donor_gk = ""
            for v, gk, _ in candidate_pool:
                if v == chosen_v:
                    donor_gk = gk
                    break
            donors[(key_s, field)] = {"value": chosen_v, "group_key": donor_gk}
            conflicts[(key_s, field)] = False

    return donors, conflicts


def compute_key_field_base_values(checks: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    base: dict[tuple[str, str], dict[str, str]] = {}
    if checks.empty:
        return base
    good = checks[checks["evidence_supported"] == 1].copy()
    good = good[good["field_name"].isin(CORE_FIELDS)]
    good["norm_v"] = good["field_value"].map(normalize_value_token)
    good = good[good["norm_v"] != ""]
    for (key, field), g in good.groupby(["key", "field_name"], sort=False):
        vc = g["norm_v"].value_counts()
        if vc.empty:
            continue
        base_val = str(vc.index[0])
        donor = g[g["norm_v"] == base_val]["group_key"].astype(str).mode()
        donor_gk = str(donor.iloc[0]) if not donor.empty else ""
        base[(str(key), str(field))] = {"base_value": base_val, "donor_group_key": donor_gk}
    return base


def discover_source_for_key(key: str, sample_manifest: pd.DataFrame, key2txt: pd.DataFrame) -> Path | None:
    key = str(key).strip()
    sub = sample_manifest[sample_manifest["key"].astype(str).str.strip() == key]
    if not sub.empty and "text_path" in sub.columns:
        raw = str(sub.iloc[0]["text_path"]).strip()
        if raw:
            p = Path(raw.replace("\\", "/"))
            if p.exists():
                return p
    if not key2txt.empty:
        sub2 = key2txt[key2txt["key"].astype(str).str.strip() == key]
        if not sub2.empty and "txt_path" in sub2.columns:
            raw2 = str(sub2.iloc[0]["txt_path"]).strip()
            if raw2:
                p2 = Path("data/cleaned/content_goren_2025") / raw2.replace("\\", "/")
                if p2.exists():
                    return p2
    return None


def build_boilerplate_lines(text: str, repeat_threshold: int, min_line_len: int) -> set[str]:
    counts: Counter[str] = Counter()
    for line in str(text).splitlines():
        n = normalize_text(line).strip()
        if len(n) < min_line_len:
            continue
        n = re.sub(r"\s+", " ", n)
        counts[n] += 1
    return {k for k, v in counts.items() if v >= repeat_threshold}


def paragraph_ranges(text: str) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    for idx, m in enumerate(re.finditer(r"(?s)\S(?:.*?\S)?(?=\n\s*\n|$)", str(text))):
        seg = m.group(0)
        ranges.append({"paragraph_id": idx, "start": m.start(), "end": m.end(), "text": seg})
    return ranges


def infer_anchor_id(
    key: str,
    span_start: str,
    evidence_text: str,
    src_text: str,
    para_ranges: list[dict[str, Any]],
) -> str:
    start = str(span_start).strip()
    if start.isdigit():
        pos = int(start)
        for p in para_ranges:
            if int(p["start"]) <= pos < int(p["end"]):
                h = hashlib.sha1(normalize_text(p["text"]).encode("utf-8", errors="ignore")).hexdigest()[:12]
                return f"{key}::p{p['paragraph_id']}::{h}"
        return f"{key}::s{start}"
    h2 = hashlib.sha1(normalize_text(evidence_text).encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{key}::e::{h2}"


def infer_field_span_start(
    key: str,
    fid: str,
    field_name: str,
    row: pd.Series,
    realign_index: dict[tuple[str, str, str], str],
) -> str:
    k = (str(key), str(fid), str(field_name))
    if k in realign_index and str(realign_index[k]).strip():
        return str(realign_index[k]).strip()
    return str(row.get("evidence_span_start", "")).strip()


def parse_numbers(text: str) -> list[float]:
    nums = []
    for tok in re.findall(r"[+-]?\d+(?:[.,]\d+)?", str(text)):
        tok2 = tok.replace(",", ".")
        try:
            nums.append(float(tok2))
        except ValueError:
            continue
    return nums


def has_ee_column_context(evidence_text: str) -> bool:
    t = normalize_text(evidence_text)
    if "encapsulation efficiency" in t:
        return True
    if "entrapment efficiency" in t:
        return True
    if "drug entrapment" in t:
        return True
    if re.search(r"\bdee\b", t):
        return True
    if re.search(r"\bee\b", t):
        return True
    return False


def numeric_anchor_pass_ee(main_numeric_token: str, evidence_text: str, rel_tol: float, abs_tol: float) -> bool:
    # EE-specific tolerant anchoring:
    # - accept rounding/format variants
    # - accept mean ± SD style contexts by matching nearby numeric values
    if not main_numeric_token:
        return False
    try:
        main_num = float(main_numeric_token)
    except ValueError:
        return False
    ev_norm = normalize_text(evidence_text)

    direct_variants = {
        main_numeric_token,
        str(main_numeric_token).replace(".", ","),
        f"{main_num:.12f}".rstrip("0").rstrip("."),
        f"{main_num:.2f}".rstrip("0").rstrip("."),
        f"{main_num:.1f}".rstrip("0").rstrip("."),
    }
    if any(v and v in ev_norm for v in direct_variants):
        return True

    ee_abs_tol = max(float(abs_tol), 0.5)
    ee_rel_tol = max(float(rel_tol), 0.02)
    for n in parse_numbers(evidence_text):
        if abs(n - main_num) <= max(ee_abs_tol, abs(main_num) * ee_rel_tol):
            return True
        if round(n, 1) == round(main_num, 1):
            return True
    return False


def numeric_anchor_pass_size(main_numeric_token: str, evidence_text: str, rel_tol: float, abs_tol: float) -> bool:
    # Size-specific tolerant anchoring:
    # - accept integer/one-decimal rounding variants
    # - accept +/- contexts by allowing practical nm tolerance
    if not main_numeric_token:
        return False
    try:
        main_num = float(main_numeric_token)
    except ValueError:
        return False
    ev_norm = normalize_text(evidence_text)

    direct_variants = {
        main_numeric_token,
        str(main_numeric_token).replace(".", ","),
        f"{main_num:.12f}".rstrip("0").rstrip("."),
        f"{main_num:.2f}".rstrip("0").rstrip("."),
        f"{main_num:.1f}".rstrip("0").rstrip("."),
        f"{round(main_num):d}",
    }
    if any(v and v in ev_norm for v in direct_variants):
        return True

    size_abs_tol = max(float(abs_tol), 1.0)
    size_rel_tol = max(float(rel_tol), 0.02)
    for n in parse_numbers(evidence_text):
        if abs(n - main_num) <= max(size_abs_tol, abs(main_num) * size_rel_tol):
            return True
        if round(n) == round(main_num):
            return True
        if round(n, 1) == round(main_num, 1):
            return True
    return False


def numeric_anchor_pass(field_name: str, main_numeric_token: str, evidence_text: str, rel_tol: float, abs_tol: float) -> bool:
    if field_name == "encapsulation_efficiency_percent":
        return numeric_anchor_pass_ee(main_numeric_token, evidence_text, rel_tol, abs_tol)
    if field_name == "size_nm":
        return numeric_anchor_pass_size(main_numeric_token, evidence_text, rel_tol, abs_tol)
    if not main_numeric_token:
        return False
    try:
        main_num = float(main_numeric_token)
    except ValueError:
        return False

    ev_norm = normalize_text(evidence_text)
    direct_variants = {
        main_numeric_token,
        str(main_numeric_token).replace(".", ","),
        f"{main_num:.12f}".rstrip("0").rstrip("."),
    }
    if any(v and v in ev_norm for v in direct_variants):
        return True

    for n in parse_numbers(evidence_text):
        if abs(n - main_num) <= max(abs_tol, abs(main_num) * rel_tol):
            return True
    return False


def infer_evidence_source_type_for_field(row: pd.Series, field_name: str) -> str:
    # Prefer explicit field-level provenance columns when present.
    field_source_map = {
        "encapsulation_efficiency_percent": "value_source_EE",
        "size_nm": "value_source_size",
        "drug_feed_amount_text": "value_source_drug_mass",
        "plga_mass_mg": "value_source_polymer_mass",
    }
    value_source_col = field_source_map.get(field_name, "")
    value_source = str(row.get(value_source_col, "")).strip().lower() if value_source_col else ""
    if value_source == "table_csv_cell":
        return "table"
    if value_source in {"fulltext_span", "proxy_compose", "derived_rule", "derived_doe_decode"}:
        return "text"

    explicit = str(row.get("evidence_source_type", "")).strip().lower()
    if explicit == "table":
        return "table"
    if explicit in {"fulltext", "text"}:
        return "text"

    method = str(row.get("evidence_method", "")).strip().lower()
    if "table" in method or "csv" in method:
        return "table"
    return "text"


def _normalize_value_for_join(v: Any) -> str:
    s = str(v or "").strip()
    if not s:
        return ""
    n = parse_first_number(s)
    if n is None:
        return normalize_value_token(s)
    return f"{n:.6f}"


def _field_to_audit_raw_col(field_name: str) -> str:
    return {
        "encapsulation_efficiency_percent": "EE_raw",
        "size_nm": "size_raw",
    }.get(field_name, "")


def _load_audit_evidence_long(audit_pack_xlsx: Path) -> pd.DataFrame:
    audit_df = pd.read_excel(audit_pack_xlsx, sheet_name="audit_cases", dtype=str).fillna("")
    if "zotero_key" in audit_df.columns and "key" not in audit_df.columns:
        audit_df["key"] = audit_df["zotero_key"].astype(str)
    out_rows: list[dict[str, str]] = []
    for _, r in audit_df.iterrows():
        key = str(r.get("key", "")).strip()
        if not key:
            continue
        for field_name in ("encapsulation_efficiency_percent", "size_nm"):
            raw_col = _field_to_audit_raw_col(field_name)
            raw_val = str(r.get(raw_col, "")).strip() if raw_col else ""
            if not raw_val:
                continue
            out_rows.append(
                {
                    "key": key,
                    "field_name": field_name,
                    "evidence_span_id": str(r.get("evidence_span_id", "")).strip(),
                    "evidence_span_start": str(r.get("evidence_span_start", "")).strip(),
                    "evidence_span_end": str(r.get("evidence_span_end", "")).strip(),
                    "extracted_value_raw": raw_val,
                    "extracted_value_norm": _normalize_value_for_join(raw_val),
                    "table_cell_text": str(r.get("table_cell_text", "")).strip(),
                    "table_row_text": str(r.get("table_row_text", "")).strip(),
                }
            )
    return pd.DataFrame(out_rows)


def _build_audit_lookup(
    audit_long: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str], list[dict[str, str]]],
    dict[tuple[str, str, str, str], list[dict[str, str]]],
    dict[tuple[str, str, str], list[dict[str, str]]],
]:
    by_span_id: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_span: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    by_value: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    if audit_long.empty:
        return by_span_id, by_span, by_value
    for _, r in audit_long.iterrows():
        rec = {k: str(v or "").strip() for k, v in r.to_dict().items()}
        key = rec.get("key", "")
        field = rec.get("field_name", "")
        sid = rec.get("evidence_span_id", "")
        ss = rec.get("evidence_span_start", "")
        se = rec.get("evidence_span_end", "")
        vnorm = rec.get("extracted_value_norm", "")
        if key and sid:
            by_span_id[(key, sid)].append(rec)
        if key and field and ss and se:
            by_span[(key, field, ss, se)].append(rec)
        if key and field and vnorm:
            by_value[(key, field, vnorm)].append(rec)
    return by_span_id, by_span, by_value


def _pick_audit_evidence_record(
    row: pd.Series,
    field_name: str,
    field_value: str,
    by_span_id: dict[tuple[str, str], list[dict[str, str]]],
    by_span: dict[tuple[str, str, str, str], list[dict[str, str]]],
    by_value: dict[tuple[str, str, str], list[dict[str, str]]],
) -> dict[str, str] | None:
    key = str(row.get("key", "")).strip()
    if not key:
        return None
    sid = str(row.get("evidence_span_id", "")).strip()
    if sid:
        c1 = by_span_id.get((key, sid), [])
        if c1:
            c1f = [x for x in c1 if x.get("field_name", "") == field_name]
            if c1f:
                return c1f[0]
    ss = str(row.get("evidence_span_start", "")).strip()
    se = str(row.get("evidence_span_end", "")).strip()
    if ss and se:
        c2 = by_span.get((key, field_name, ss, se), [])
        if len(c2) == 1:
            return c2[0]
        if len(c2) > 1:
            val_norm = _normalize_value_for_join(field_value)
            c2v = [x for x in c2 if x.get("extracted_value_norm", "") == val_norm]
            if len(c2v) == 1:
                return c2v[0]
    val_norm = _normalize_value_for_join(field_value)
    if not val_norm:
        return None
    c3 = by_value.get((key, field_name, val_norm), [])
    if len(c3) == 1:
        return c3[0]
    return None


def is_deprioritized_section(row: pd.Series, evidence_text: str) -> bool:
    section = normalize_text(row.get("evidence_section", ""))
    if section in DEPRIORITIZED_SECTION_HINTS:
        return True
    e = normalize_text(evidence_text)
    return any(h in e for h in ("abstract", "introduction", "background"))


def evidence_contains_boilerplate(evidence_text: str, boilerplate_lines: set[str], min_line_len: int) -> bool:
    if not boilerplate_lines:
        return False
    for line in str(evidence_text).splitlines():
        n = re.sub(r"\s+", " ", normalize_text(line).strip())
        if len(n) < min_line_len:
            continue
        if n in boilerplate_lines:
            return True
    return False


def build_realign_index(realign_log_path: Path) -> dict[tuple[str, str, str], str]:
    out: dict[tuple[str, str, str], str] = {}
    if not realign_log_path.exists():
        return out
    rl = pd.read_csv(realign_log_path, sep="\t", dtype=str).fillna("")
    for _, r in rl.iterrows():
        if str(r.get("realign_success", "")).lower() not in {"true", "1"}:
            continue
        key = str(r.get("key", "")).strip()
        fid = str(r.get("formulation_id", "")).strip()
        field = str(r.get("field_name", "")).strip()
        start = str(r.get("new_span_start", "")).strip()
        if key and fid and field and start:
            out[(key, fid, field)] = start
    return out


def run_qc(
    df: pd.DataFrame,
    args: argparse.Namespace,
    sample_manifest: pd.DataFrame,
    key2txt: pd.DataFrame,
    realign_index: dict[tuple[str, str, str], str],
    audit_by_span_id: dict[tuple[str, str], list[dict[str, str]]] | None = None,
    audit_by_span: dict[tuple[str, str, str, str], list[dict[str, str]]] | None = None,
    audit_by_value: dict[tuple[str, str, str], list[dict[str, str]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    doc_cache: dict[str, dict[str, Any]] = {}

    def get_doc_context(key: str) -> dict[str, Any]:
        if key in doc_cache:
            return doc_cache[key]
        src = discover_source_for_key(key, sample_manifest, key2txt)
        if src is None:
            ctx = {"source_text": "", "boilerplate_lines": set(), "paragraphs": []}
        else:
            txt = src.read_text(encoding="utf-8", errors="ignore")
            ctx = {
                "source_text": txt,
                "boilerplate_lines": build_boilerplate_lines(
                    txt,
                    repeat_threshold=int(args.boilerplate_repeat_threshold),
                    min_line_len=int(args.boilerplate_min_line_len),
                ),
                "paragraphs": paragraph_ranges(txt),
            }
        doc_cache[key] = ctx
        return ctx

    rows: list[dict[str, Any]] = []
    audit_by_span_id = audit_by_span_id or {}
    audit_by_span = audit_by_span or {}
    audit_by_value = audit_by_value or {}
    for idx, r in df.iterrows():
        base_evidence_text = str(r.get("evidence_span_text", ""))
        evidence = normalize_text(base_evidence_text)
        if not evidence.strip():
            continue
        key = str(r.get("key", "")).strip()
        fid = str(r.get("formulation_id", "")).strip()
        group_key = f"{key}::{fid}"
        doc_ctx = get_doc_context(key)
        boilerplate_hit = evidence_contains_boilerplate(
            str(r.get("evidence_span_text", "")),
            doc_ctx["boilerplate_lines"],
            int(args.boilerplate_min_line_len),
        )
        deprioritized = is_deprioritized_section(r, str(r.get("evidence_span_text", "")))

        for field_name, spec in FIELD_SPECS.items():
            if field_name not in df.columns:
                continue
            value_text = str(r.get(field_name, "")).strip()
            if not value_text:
                continue
            numeric_tokens = find_numeric_tokens(value_text)
            if not numeric_tokens:
                continue
            main_numeric = numeric_tokens[0]

            unit_tokens = build_required_unit_tokens(field_name, value_text)
            entity_tokens = build_required_entity_tokens(field_name, r)
            field_evidence_text = base_evidence_text
            evidence_used_source = "text"
            if field_name in {"encapsulation_efficiency_percent", "size_nm"} and audit_by_span_id:
                picked = _pick_audit_evidence_record(
                    row=r,
                    field_name=field_name,
                    field_value=value_text,
                    by_span_id=audit_by_span_id,
                    by_span=audit_by_span,
                    by_value=audit_by_value,
                )
                if picked is not None:
                    table_cell = str(picked.get("table_cell_text", "")).strip()
                    table_row = str(picked.get("table_row_text", "")).strip()
                    if table_cell:
                        field_evidence_text = table_cell
                        evidence_used_source = "table_cell"
                    elif table_row:
                        field_evidence_text = table_row
                        evidence_used_source = "table_row"

            field_evidence_norm = normalize_text(field_evidence_text)
            has_numeric = numeric_anchor_pass(
                field_name,
                main_numeric,
                field_evidence_text,
                rel_tol=float(args.numeric_rel_tol),
                abs_tol=float(args.numeric_abs_tol),
            )
            unit_matches = contains_any(field_evidence_norm, unit_tokens)
            # EE can appear without explicit '%' if the span is clearly in EE column/context.
            if field_name == "encapsulation_efficiency_percent" and has_numeric and unit_matches == 0:
                if has_ee_column_context(field_evidence_text):
                    unit_matches = 1
            entity_matches = contains_any(field_evidence_norm, entity_tokens)

            fail_numeric = not has_numeric
            fail_unit = bool(args.require_unit_token and unit_matches < int(args.unit_min_matches))
            require_entity = (
                args.require_entity_for_mass and field_name in {"drug_feed_amount_text", "plga_mass_mg"} and len(entity_tokens) > 0
            )
            fail_entity = bool(require_entity and entity_matches < int(args.entity_min_matches))
            fail_boilerplate = bool(boilerplate_hit)
            fail_deprioritized_section = bool(deprioritized and not has_numeric)
            mismatch = bool(fail_numeric or fail_unit or fail_entity or fail_boilerplate or fail_deprioritized_section)

            span_start = infer_field_span_start(key, fid, field_name, r, realign_index)
            anchor_id = infer_anchor_id(
                key=key,
                span_start=span_start,
                evidence_text=str(r.get("evidence_span_text", "")),
                src_text=doc_ctx["source_text"],
                para_ranges=doc_ctx["paragraphs"],
            )
            ee_header_in_span = 0
            ee_header_in_window = 0
            ee_header_in_line_window = 0
            ee_table_chunk_is_table_like = 0
            ee_table_context_pass = 0
            ee_window_snippet = ""
            if field_name == "encapsulation_efficiency_percent":
                ev_txt = field_evidence_text
                win = get_char_window(doc_ctx["source_text"], span_start, int(args.ee_window_chars))
                line_win = get_line_window(doc_ctx["source_text"], span_start, half_lines=25)
                ee_header_in_span = int(has_ee_header_hint(ev_txt))
                ee_header_in_window = int(has_ee_header_hint(win))
                ee_header_in_line_window = int(has_ee_header_hint(line_win))
                ee_table_chunk_is_table_like = int(is_table_like_chunk(line_win))
                if args.ee_structured_mode == "legacy":
                    ee_table_context_pass = int(ee_header_in_span == 1)
                else:
                    ee_table_context_pass = int(
                        (ee_header_in_span == 1)
                        or (ee_header_in_line_window == 1 and ee_table_chunk_is_table_like == 1)
                        or (ee_header_in_window == 1 and ee_table_chunk_is_table_like == 1)
                    )
                ee_window_snippet = short_text(line_win if line_win else win, 220)

            rows.append(
                {
                    "row_index": int(idx),
                    "key": key,
                    "formulation_id": fid,
                    "group_key": group_key,
                    "field_name": field_name,
                    "field_value": value_text,
                    "main_numeric_token": main_numeric,
                    "required_unit_tokens": "|".join(unit_tokens),
                    "required_entity_tokens": "|".join(entity_tokens),
                    "has_main_numeric_token": int(has_numeric),
                    "unit_match_count": int(unit_matches),
                    "entity_match_count": int(entity_matches),
                    "fail_numeric_token": int(fail_numeric),
                    "fail_unit_token": int(fail_unit),
                    "fail_entity_token": int(fail_entity),
                    "fail_boilerplate": int(fail_boilerplate),
                    "fail_deprioritized_section": int(fail_deprioritized_section),
                    "evidence_mismatch": int(mismatch),
                    "evidence_supported": int(not mismatch),
                    "field_span_start": span_start,
                    "anchor_id": anchor_id,
                    "evidence_span_text": short_text(field_evidence_text, 300),
                    "drug_name": str(r.get("drug_name", "")).strip(),
                    "ee_header_in_span": int(ee_header_in_span),
                    "ee_header_in_window": int(ee_header_in_window),
                    "ee_header_in_line_window": int(ee_header_in_line_window),
                    "ee_table_chunk_is_table_like": int(ee_table_chunk_is_table_like),
                    "ee_table_context_pass": int(ee_table_context_pass),
                    "ee_window_snippet": ee_window_snippet,
                    "evidence_source_type": infer_evidence_source_type_for_field(r, field_name),
                    "evidence_used_source": evidence_used_source,
                }
            )

    checks = pd.DataFrame(rows)
    if checks.empty:
        checks = pd.DataFrame(
            columns=[
                "row_index",
                "key",
                "formulation_id",
                "group_key",
                "field_name",
                "field_value",
                "main_numeric_token",
                "required_unit_tokens",
                "required_entity_tokens",
                "has_main_numeric_token",
                "unit_match_count",
                "entity_match_count",
                "fail_numeric_token",
                "fail_unit_token",
                "fail_entity_token",
                "fail_boilerplate",
                "fail_deprioritized_section",
                "evidence_mismatch",
                "evidence_supported",
                "field_span_start",
                "anchor_id",
                "evidence_span_text",
                "drug_name",
                "evidence_source_type",
                "evidence_used_source",
            ]
        )
    flagged = checks[checks["evidence_mismatch"] == 1].copy()

    report_rows: list[dict[str, Any]] = []
    for fn, g in checks.groupby("field_name", sort=True):
        n = int(len(g))
        m = int((g["evidence_mismatch"] == 1).sum())
        report_rows.append(
            {
                "field_name": fn,
                "evaluated_rows": n,
                "mismatch_rows": m,
                "mismatch_rate": (m / n) if n else 0.0,
                "fail_numeric_token_rows": int((g["fail_numeric_token"] == 1).sum()),
                "fail_unit_token_rows": int((g["fail_unit_token"] == 1).sum()),
                "fail_entity_token_rows": int((g["fail_entity_token"] == 1).sum()),
                "fail_boilerplate_rows": int((g["fail_boilerplate"] == 1).sum()),
                "fail_deprioritized_section_rows": int((g["fail_deprioritized_section"] == 1).sum()),
            }
        )
    report = pd.DataFrame(report_rows).sort_values("mismatch_rate", ascending=False).reset_index(drop=True)

    bad_row_indices = set(flagged["row_index"].astype(int).tolist())
    high_conf = df.copy()
    high_conf["row_index"] = range(len(high_conf))
    high_conf["evidence_mismatch_any"] = high_conf["row_index"].map(lambda i: 1 if int(i) in bad_row_indices else 0)
    high_conf = high_conf[high_conf["evidence_mismatch_any"] == 0].reset_index(drop=True)

    risk = build_shared_span_risk_flags(checks)
    confidence = build_formulation_confidence(checks, risk, df)
    return checks, report, flagged, high_conf, risk, confidence


def build_shared_span_risk_flags(checks: pd.DataFrame) -> pd.DataFrame:
    if checks.empty:
        return pd.DataFrame(
            columns=[
                "key",
                "field_name",
                "anchor_id",
                "n_formulation_rows_reusing_span",
                "group_keys",
                "row_indices",
            ]
        )
    reuse = (
        checks.groupby(["key", "field_name", "anchor_id"], dropna=False)
        .agg(
            n_formulation_rows_reusing_span=("group_key", "nunique"),
            group_keys=("group_key", lambda x: "|".join(sorted(set(x)))),
            row_indices=("row_index", lambda x: "|".join(str(int(i)) for i in sorted(set(x)))),
        )
        .reset_index()
    )
    reuse = reuse[reuse["n_formulation_rows_reusing_span"] > 1].sort_values(
        ["n_formulation_rows_reusing_span", "key", "field_name"],
        ascending=[False, True, True],
    )
    return reuse.reset_index(drop=True)


def build_formulation_confidence(
    checks: pd.DataFrame,
    risk_flags: pd.DataFrame,
    all_rows: pd.DataFrame,
) -> pd.DataFrame:
    if checks.empty and all_rows.empty:
        return pd.DataFrame(
            columns=[
                "key",
                "formulation_id",
                "group_key",
                "condition_instance_key",
                "condition_instance_source",
                "has_explicit_formulation_id",
                "n_core_fields_supported",
                "strong_local_support",
                "ee_structured_support",
                "n_fields_local_evidence",
                "n_fields_inherited_base",
                "ee_local_support",
                "ee_support_level",
                "ee_fail_reason",
                "ee_evidence_snippet",
                "ee_evidence_block_id",
                "loading_proxy_supported",
                "loading_proxy_support_path",
                "polymer_identity_supported",
                "polymer_identity_support_level",
                "fingerprint_field_count_present",
                "fingerprint_field_count_total",
                "fingerprint_completeness",
                "has_explicit_constant_parameter_claim",
                "inheritance_without_constant_claim",
                "evidence_source_type_by_field_json",
                "inherited_base_donor_by_field_json",
                "shared_span_risk_fields_count",
                "confidence_tier",
            ]
        )

    risk_map: dict[tuple[str, str], set[str]] = {}
    if not risk_flags.empty:
        for _, rr in risk_flags.iterrows():
            key = str(rr.get("key", "")).strip()
            field = str(rr.get("field_name", "")).strip()
            gks = str(rr.get("group_keys", "")).split("|")
            for gk in gks:
                gk = gk.strip()
                if not gk:
                    continue
                risk_map.setdefault((key, gk), set()).add(field)

    if checks.empty:
        core = pd.DataFrame(columns=["key", "formulation_id", "group_key", "field_name", "evidence_supported", "anchor_id"])
    else:
        core = checks[
            checks["field_name"].isin(CORE_FIELDS)
        ][
            [
                "key",
                "formulation_id",
                "group_key",
                "field_name",
                "evidence_supported",
                "field_value",
            ]
        ].copy()
        core["evidence_supported"] = core["evidence_supported"].astype(int)

    core_local_support: dict[tuple[str, str, str], set[str]] = {}
    if not core.empty:
        for (key, fid, gk), g in core.groupby(["key", "formulation_id", "group_key"], sort=False):
            s = set(g[g["evidence_supported"] == 1]["field_name"].astype(str).tolist())
            core_local_support[(str(key), str(fid), str(gk))] = s

    constant_claim_by_key = compute_key_level_constant_claims(all_rows)
    base_values = compute_key_field_base_values(core if not core.empty else pd.DataFrame())
    doc_polymer_donors, doc_polymer_conflicts = choose_doc_polymer_donors(all_rows, checks)

    group_blob_map: dict[str, str] = {}
    if not checks.empty and "group_key" in checks.columns and "evidence_span_text" in checks.columns:
        for gk, gg in checks.groupby("group_key", sort=False):
            group_blob_map[str(gk)] = "\n".join(gg["evidence_span_text"].astype(str).tolist())

    all_formulations = (
        all_rows[["key", "formulation_id"]]
        .drop_duplicates()
        .assign(group_key=lambda d: d["key"].astype(str).str.strip() + "::" + d["formulation_id"].astype(str).str.strip())
    )
    row_indexed = all_rows.copy()
    row_indexed["_group_key"] = row_indexed["key"].astype(str).str.strip() + "::" + row_indexed["formulation_id"].astype(str).str.strip()
    repr_rows = row_indexed.drop_duplicates(subset=["_group_key"], keep="first").set_index("_group_key", drop=False)

    out_rows: list[dict[str, Any]] = []
    grouped = {
        (str(k), str(fid), str(gk)): g for (k, fid, gk), g in core.groupby(["key", "formulation_id", "group_key"], sort=True)
    }
    for _, base in all_formulations.iterrows():
        key = str(base["key"])
        fid = str(base["formulation_id"])
        gk = str(base["group_key"])
        g = grouped.get((key, fid, gk), pd.DataFrame(columns=core.columns))
        supp = set(g[g["evidence_supported"] == 1]["field_name"].astype(str).tolist())
        n_supported = int(len(supp))

        risk_fields_count = len(risk_map.get((str(key), str(gk)), set()))
        row_data = repr_rows.loc[gk] if gk in repr_rows.index else pd.Series(dtype=str)
        condition_instance_key, condition_instance_source = build_condition_instance_key(row_data if not row_data.empty else pd.Series({"key": key, "formulation_id": fid}))
        has_explicit_id = int(is_explicit_formulation_id(fid))
        has_constant_claim = bool(constant_claim_by_key.get(key, False))

        # Per-field local vs inherited_base typing.
        source_by_field: dict[str, str] = {}
        donor_by_field: dict[str, str] = {}
        n_local = 0
        n_inherited = 0
        local_supported_fields = core_local_support.get((key, fid, gk), set())
        for field in CORE_FIELDS:
            row_val = normalize_value_token(row_data.get(field, "") if not row_data.empty else "")
            if field in local_supported_fields and row_val:
                source_by_field[field] = "local"
                donor_by_field[field] = ""
                n_local += 1
            elif row_val:
                # Present but unsupported still treated as local extraction.
                source_by_field[field] = "local"
                donor_by_field[field] = ""
            else:
                b = None
                if field in {"la_ga_ratio", "polymer_mw_kDa"}:
                    if not doc_polymer_conflicts.get((key, field), False):
                        d = doc_polymer_donors.get((key, field), {})
                        if d.get("value"):
                            b = {"base_value": d.get("value", ""), "donor_group_key": d.get("group_key", "")}
                if b is None:
                    b = base_values.get((key, field), {})
                if b.get("base_value"):
                    source_by_field[field] = "inherited_base"
                    donor_by_field[field] = str(b.get("donor_group_key", ""))
                    n_inherited += 1
                else:
                    source_by_field[field] = "local"
                    donor_by_field[field] = ""

        inheritance_without_claim = bool(n_inherited > 0 and not has_constant_claim)
        strong_local_support = int("encapsulation_efficiency_percent" in local_supported_fields)
        ee_structured_support = 0
        ee_support_level = "none"
        ee_fail_reason = "other"
        ee_evidence_snippet = short_text(str(row_data.get("evidence_span_text", "")), 120)
        ee_evidence_block_id = ""
        ee_rows = checks[(checks["group_key"] == gk) & (checks["field_name"] == "encapsulation_efficiency_percent")] if not checks.empty else pd.DataFrame()
        ee_candidate_num = parse_first_number(str(row_data.get("encapsulation_efficiency_percent", "")) if not row_data.empty else "")
        ee_has_candidate_value = ee_candidate_num is not None
        appears_in_ee_only_view = bool(str(row_data.get("encapsulation_efficiency_percent", "")).strip()) if not row_data.empty else False
        is_table_derived_instance = condition_instance_source in {"table_row_conditions", "explicit_formulation_id"}
        if not ee_rows.empty:
            ee_evidence_snippet = short_text(str(ee_rows.iloc[0].get("evidence_span_text", "")), 120)
            ee_evidence_block_id = str(ee_rows.iloc[0].get("anchor_id", "")).strip()

        ee_context_pass = int(
            (not ee_rows.empty) and (pd.to_numeric(ee_rows.get("ee_table_context_pass", 0), errors="coerce").fillna(0).astype(int).max() >= 1)
        )
        ee_any_boilerplate = int(
            (not ee_rows.empty) and (pd.to_numeric(ee_rows.get("fail_boilerplate", 0), errors="coerce").fillna(0).astype(int).max() >= 1)
        )
        ee_any_section_filtered = int(
            (not ee_rows.empty) and (pd.to_numeric(ee_rows.get("fail_deprioritized_section", 0), errors="coerce").fillna(0).astype(int).max() >= 1)
        )
        ee_any_numeric_fail = int(
            (not ee_rows.empty) and (pd.to_numeric(ee_rows.get("fail_numeric_token", 0), errors="coerce").fillna(0).astype(int).max() >= 1)
        )

        if strong_local_support == 1:
            ee_support_level = "strong_local"
            ee_fail_reason = ""
        else:
            if ee_has_candidate_value and (is_table_derived_instance or appears_in_ee_only_view) and (ee_context_pass == 1):
                ee_structured_support = 1
                ee_support_level = "structured_table"
                ee_fail_reason = ""
            else:
                ee_structured_support = 0
                ee_support_level = "none"
                if ee_has_candidate_value and ee_any_numeric_fail == 1:
                    ee_fail_reason = "numeric_anchor_fail"
                elif ee_has_candidate_value and ee_context_pass == 0:
                    ee_fail_reason = "header_context_missing"
                elif ee_has_candidate_value and ee_any_section_filtered == 1:
                    ee_fail_reason = "section_filtered"
                elif ee_has_candidate_value and ee_any_boilerplate == 1:
                    ee_fail_reason = "other"
                elif not ee_has_candidate_value or ee_rows.empty:
                    ee_fail_reason = "missing_span"
                else:
                    ee_fail_reason = "other"
        ee_local_support = int((strong_local_support == 1) or (ee_structured_support == 1))

        # Loading proxy support: local + structured support paths.
        lc_local = int("loading_content_percent" in local_supported_fields)
        ratio_local = int(("drug_feed_amount_text" in local_supported_fields) and ("plga_mass_mg" in local_supported_fields))
        row_blob = (
            f"{str(row_data.get('notes', ''))}\n"
            f"{str(row_data.get('evidence_span_text', ''))}\n"
            f"{group_blob_map.get(gk, '')}"
        )
        lc_structured = 0
        if condition_instance_source == "table_row_conditions":
            lc_val = str(row_data.get("loading_content_percent", "")).strip() if not row_data.empty else ""
            if lc_val and has_loading_header_hint(row_blob):
                lc_structured = 1

        ratio_structured = 0
        dval = str(row_data.get("drug_feed_amount_text", "")).strip() if not row_data.empty else ""
        pval = str(row_data.get("plga_mass_mg", "")).strip() if not row_data.empty else ""
        if dval and pval and has_ratio_header_hint(row_blob):
            ratio_structured = 1

        loading_proxy_supported = int((lc_local == 1) or (ratio_local == 1) or (lc_structured == 1) or (ratio_structured == 1))
        loading_paths = []
        if lc_local == 1:
            loading_paths.append("LC_local")
        if ratio_local == 1:
            loading_paths.append("ratio_local")
        if lc_structured == 1:
            loading_paths.append("LC_structured")
        if ratio_structured == 1:
            loading_paths.append("ratio_structured")
        loading_proxy_support_path = "|".join(loading_paths) if loading_paths else "none"

        polymer_local = int(
            ("la_ga_ratio" in local_supported_fields) or ("polymer_mw_kDa" in local_supported_fields)
        )
        polymer_inherited = int(
            (source_by_field.get("la_ga_ratio", "local") == "inherited_base")
            or (source_by_field.get("polymer_mw_kDa", "local") == "inherited_base")
        )
        polymer_identity_supported = int((polymer_local == 1) or (polymer_inherited == 1))
        if polymer_local == 1:
            polymer_identity_support_level = "strong_local"
        elif polymer_inherited == 1:
            polymer_identity_support_level = "inherited_global"
        else:
            polymer_identity_support_level = "none"

        # Fingerprint completeness from available condition dimensions.
        fp_present = 0
        for f in FINGERPRINT_FIELDS:
            if f in row_data.index and str(row_data.get(f, "")).strip():
                fp_present += 1
        fp_total = int(len(FINGERPRINT_FIELDS))
        fp_completeness = float(fp_present / fp_total) if fp_total else 0.0

        tier = "C"
        if ee_local_support == 1 and fp_completeness >= 0.45 and n_local >= 3:
            tier = "A"
        elif ee_local_support == 1 and fp_completeness >= 0.25 and (n_local + n_inherited) >= 2:
            tier = "B"

        if inheritance_without_claim and tier == "A":
            tier = "B"
        elif inheritance_without_claim and tier == "B":
            tier = "C"
        if risk_fields_count >= 6 and tier == "A":
            tier = "B"
        if risk_fields_count >= 8 and tier == "B":
            tier = "C"

        row_out = {
            "key": str(key),
            "formulation_id": str(fid),
            "group_key": str(gk),
            "condition_instance_key": condition_instance_key,
            "condition_instance_source": condition_instance_source,
            "has_explicit_formulation_id": has_explicit_id,
            "n_core_fields_supported": n_supported,
            "strong_local_support": int(strong_local_support),
            "ee_structured_support": int(ee_structured_support),
            "n_fields_local_evidence": int(n_local),
            "n_fields_inherited_base": int(n_inherited),
            "ee_local_support": int(ee_local_support),
            "ee_support_level": ee_support_level,
            "ee_fail_reason": ee_fail_reason,
            "ee_evidence_snippet": ee_evidence_snippet,
            "ee_evidence_block_id": ee_evidence_block_id,
            "loading_proxy_supported": int(loading_proxy_supported),
            "loading_proxy_support_path": loading_proxy_support_path,
            "polymer_identity_supported": int(polymer_identity_supported),
            "polymer_identity_support_level": polymer_identity_support_level,
            "fingerprint_field_count_present": int(fp_present),
            "fingerprint_field_count_total": int(fp_total),
            "fingerprint_completeness": round(fp_completeness, 6),
            "has_explicit_constant_parameter_claim": int(has_constant_claim),
            "inheritance_without_constant_claim": int(inheritance_without_claim),
            "evidence_source_type_by_field_json": json.dumps(source_by_field, ensure_ascii=False, sort_keys=True),
            "inherited_base_donor_by_field_json": json.dumps(donor_by_field, ensure_ascii=False, sort_keys=True),
            "shared_span_risk_fields_count": int(risk_fields_count),
            "confidence_tier": tier,
        }
        for field in CORE_FIELDS:
            row_out[f"evidence_source_type__{field}"] = source_by_field.get(field, "local")
        out_rows.append(row_out)

    return pd.DataFrame(out_rows).sort_values(["key", "formulation_id"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    input_tsv = Path(args.input_tsv) if args.input_tsv else Path(f"data/results/{run_id}/weak_labels__gemini.tsv")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/results/{run_id}/benchmark_goren_2025/derivation_v1")
    run_dir = Path(f"data/results/{run_id}")
    realign_log = Path(args.realign_log) if args.realign_log else run_dir / "evidence_realign_log.tsv"
    sample_manifest_path = Path(args.sample_manifest)
    key2txt_path = Path(args.key2txt)
    audit_pack_xlsx = Path(args.audit_pack_xlsx) if args.audit_pack_xlsx else None
    out_dir.mkdir(parents=True, exist_ok=True)
    if not input_tsv.exists():
        raise FileNotFoundError(f"Missing input extraction TSV: {input_tsv}")
    if not sample_manifest_path.exists():
        raise FileNotFoundError(f"Missing sample manifest TSV: {sample_manifest_path}")

    df = canonicalize_input_columns(pd.read_csv(input_tsv, sep="\t", dtype=str).fillna(""))
    if "evidence_span_text" not in df.columns:
        raise RuntimeError("Input TSV missing required column: evidence_span_text")
    sample_manifest = pd.read_csv(sample_manifest_path, sep="\t", dtype=str).fillna("")
    key2txt = pd.read_csv(key2txt_path, sep="\t", dtype=str).fillna("") if key2txt_path.exists() else pd.DataFrame()
    realign_index = build_realign_index(realign_log)
    audit_by_span_id: dict[tuple[str, str], list[dict[str, str]]] = {}
    audit_by_span: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    audit_by_value: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    if audit_pack_xlsx is not None:
        if not audit_pack_xlsx.exists():
            raise FileNotFoundError(f"Missing audit pack workbook: {audit_pack_xlsx}")
        audit_long = _load_audit_evidence_long(audit_pack_xlsx)
        audit_by_span_id, audit_by_span, audit_by_value = _build_audit_lookup(audit_long)

    checks, report, flagged, high_conf, risk_flags, confidence = run_qc(
        df,
        args,
        sample_manifest,
        key2txt,
        realign_index,
        audit_by_span_id=audit_by_span_id,
        audit_by_span=audit_by_span,
        audit_by_value=audit_by_value,
    )

    p_report = out_dir / "qc_report.tsv"
    p_flagged = out_dir / "flagged_rows_for_review.tsv"
    p_high = out_dir / "weak_labels__gemini_high_confidence.tsv"
    p_checks = out_dir / "evidence_token_qc_checks.tsv"
    p_checks_realigned = out_dir / "evidence_token_qc_checks__realigned.tsv"
    p_numeric_breakdown = out_dir / "numeric_token_mismatch_by_evidence_source_type.tsv"
    p_numeric_used_source = out_dir / "numeric_token_mismatch_by_evidence_used_source.tsv"
    p_run_summary = run_dir / "qc_field_evidence_gate_summary__realigned.tsv"
    p_risk = run_dir / "risk_flags__shared_spans.tsv"
    p_conf = run_dir / "confidence_tiers__formulation_level.tsv"

    report.to_csv(p_report, sep="\t", index=False)
    flagged.to_csv(p_flagged, sep="\t", index=False)
    high_conf.to_csv(p_high, sep="\t", index=False)
    checks.to_csv(p_checks, sep="\t", index=False)
    checks.to_csv(p_checks_realigned, sep="\t", index=False)
    if checks.empty:
        pd.DataFrame(
            columns=["field_name", "evidence_source_type", "numeric_token_mismatch_rows", "evaluated_rows"]
        ).to_csv(p_numeric_breakdown, sep="\t", index=False)
        pd.DataFrame(
            columns=["field_name", "evidence_used_source", "numeric_token_mismatch_rows", "evaluated_rows"]
        ).to_csv(p_numeric_used_source, sep="\t", index=False)
    else:
        mismatch_source = (
            checks.groupby(["field_name", "evidence_source_type"], dropna=False)
            .agg(
                numeric_token_mismatch_rows=(
                    "fail_numeric_token",
                    lambda x: int((pd.to_numeric(x, errors="coerce").fillna(0).astype(int) == 1).sum()),
                ),
                evaluated_rows=("row_index", "count"),
            )
            .reset_index()
            .sort_values(["numeric_token_mismatch_rows", "field_name", "evidence_source_type"], ascending=[False, True, True])
        )
        mismatch_source.to_csv(p_numeric_breakdown, sep="\t", index=False)
        mismatch_used_source = (
            checks.groupby(["field_name", "evidence_used_source"], dropna=False)
            .agg(
                numeric_token_mismatch_rows=(
                    "fail_numeric_token",
                    lambda x: int((pd.to_numeric(x, errors="coerce").fillna(0).astype(int) == 1).sum()),
                ),
                evaluated_rows=("row_index", "count"),
            )
            .reset_index()
            .sort_values(["numeric_token_mismatch_rows", "field_name", "evidence_used_source"], ascending=[False, True, True])
        )
        mismatch_used_source.to_csv(p_numeric_used_source, sep="\t", index=False)
    report.rename(
        columns={
            "evaluated_rows": "total_extracted",
            "mismatch_rows": "evidence_fail",
            "mismatch_rate": "fail_rate",
        }
    ).assign(
        evidence_pass=lambda x: x["total_extracted"] - x["evidence_fail"]
    )[
        [
            "field_name",
            "total_extracted",
            "evidence_pass",
            "evidence_fail",
            "fail_rate",
        ]
    ].to_csv(p_run_summary, sep="\t", index=False)

    risk_flags.to_csv(p_risk, sep="\t", index=False)
    confidence.to_csv(p_conf, sep="\t", index=False)

    summary = {
        "input_rows": int(len(df)),
        "qc_evaluated_field_rows": int(sum(report["evaluated_rows"])) if not report.empty else 0,
        "qc_mismatch_field_rows": int(sum(report["mismatch_rows"])) if not report.empty else 0,
        "flagged_rows_for_review": int(len(flagged)),
        "high_confidence_rows": int(len(high_conf)),
        "shared_span_risk_flags": int(len(risk_flags)),
        "confidence_rows": int(len(confidence)),
        "outputs": {
            "qc_report_tsv": str(p_report),
            "flagged_rows_for_review_tsv": str(p_flagged),
            "high_confidence_tsv": str(p_high),
            "checks_tsv": str(p_checks),
            "checks_realigned_tsv": str(p_checks_realigned),
            "numeric_mismatch_by_source_tsv": str(p_numeric_breakdown),
            "numeric_mismatch_by_evidence_used_source_tsv": str(p_numeric_used_source),
            "run_level_qc_summary_tsv": str(p_run_summary),
            "risk_flags_shared_spans_tsv": str(p_risk),
            "confidence_tiers_formulation_level_tsv": str(p_conf),
        },
        "params": {
            "unit_min_matches": int(args.unit_min_matches),
            "entity_min_matches": int(args.entity_min_matches),
            "require_unit_token": bool(args.require_unit_token),
            "require_entity_for_mass": bool(args.require_entity_for_mass),
            "boilerplate_repeat_threshold": int(args.boilerplate_repeat_threshold),
            "boilerplate_min_line_len": int(args.boilerplate_min_line_len),
            "numeric_rel_tol": float(args.numeric_rel_tol),
            "numeric_abs_tol": float(args.numeric_abs_tol),
            "ee_structured_mode": str(args.ee_structured_mode),
            "ee_window_chars": int(args.ee_window_chars),
            "audit_pack_xlsx": str(audit_pack_xlsx) if audit_pack_xlsx is not None else "",
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
