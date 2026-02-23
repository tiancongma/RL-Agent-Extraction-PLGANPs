#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"
DEFAULT_MAX_CODE_ABS = 3.0
DEFAULT_MAX_CODE_UNIQUE = 9
DEFAULT_CODE_MATCH_TOL = 1e-6
DEFAULT_MIN_DECODED_RATE = 0.8


@dataclass(frozen=True)
class TableBlock:
    table_index: int
    header: list[str]
    rows: list[list[str]]
    caption: str
    source_anchor: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive DOE coded/decoded factor rows by reading original HTML/TXT source tables."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--input-tsv", default="")
    parser.add_argument("--derived-tsv", default="")
    parser.add_argument("--sample-manifest", default="data/cleaned/samples/sample_goren18.tsv")
    parser.add_argument("--key2txt", default="data/cleaned/content_goren_2025/key2txt.tsv")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--max-code-abs", type=float, default=DEFAULT_MAX_CODE_ABS)
    parser.add_argument("--max-code-unique", type=int, default=DEFAULT_MAX_CODE_UNIQUE)
    parser.add_argument("--code-match-tol", type=float, default=DEFAULT_CODE_MATCH_TOL)
    parser.add_argument("--min-decoded-rate", type=float, default=DEFAULT_MIN_DECODED_RATE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_doi(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"^doi\s*:\s*", "", s)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\.org/", "", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[)\],.;]+$", "", s)
    return s


def normalize_minus(value: str) -> str:
    return (
        str(value)
        .replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2010", "-")
    )


def canonicalize_numeric_string(num: float) -> str:
    if abs(num) < 1e-12:
        return "0"
    text = f"{num:.12f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", "+0"} else text


def normalize_factor_name(value: str) -> str:
    s = normalize_minus(value).strip().lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def parse_code_value(value: str) -> tuple[str, float | None, str | None]:
    s = normalize_minus(str(value).strip())
    if not s:
        return ("", None, None)
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
        return (s, None, None)
    try:
        num = float(s)
    except ValueError:
        return (s, None, None)
    return (s, num, canonicalize_numeric_string(num))


def analyze_discrete_code_column(
    values: list[str],
    *,
    max_code_abs: float,
    max_unique: int,
    require_repeated: bool = True,
) -> tuple[bool, float, int]:
    if not values:
        return (False, 0.0, 0)
    parsed = [parse_code_value(v) for v in values]
    nums = [num for _, num, _ in parsed if num is not None]
    canons = [canon for _, _, canon in parsed if canon is not None]
    if not nums:
        return (False, 0.0, 0)
    parse_ratio = len(nums) / len(values)
    uniq = len(set(canons))
    max_abs = max(abs(n) for n in nums) if nums else 0.0
    repeated = uniq < len(nums)
    repeat_ok = repeated if require_repeated else True
    ok = parse_ratio >= 0.75 and uniq <= max_unique and repeat_ok and max_abs <= max_code_abs
    return (ok, parse_ratio, uniq)

def parse_first_float(value: str) -> float | None:
    s = normalize_minus(str(value).replace(",", "").strip())
    m = re.search(r"[+-]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_factor_label(label: str) -> tuple[str, str]:
    raw = str(label).strip()
    unit = ""
    m = re.search(r"\(([^)]{1,30})\)", raw)
    if m:
        unit = m.group(1).strip()
    return raw, unit


def infer_doi_map(extracted: pd.DataFrame, sample_manifest_path: Path) -> dict[str, str]:
    key_to_doi: dict[str, str] = {}
    if sample_manifest_path.exists():
        sm = pd.read_csv(sample_manifest_path, sep="\t", dtype=str).fillna("")
        if "key" in sm.columns and "doi" in sm.columns:
            for _, row in sm.iterrows():
                key_to_doi[str(row["key"]).strip()] = normalize_doi(row["doi"])
    return key_to_doi


def discover_source_for_key(
    key: str,
    sample_manifest_path: Path,
    key2txt_path: Path,
) -> tuple[str, Path | None]:
    key = str(key).strip()
    if sample_manifest_path.exists():
        sm = pd.read_csv(sample_manifest_path, sep="\t", dtype=str).fillna("")
        sub = sm[sm["key"].astype(str).str.strip() == key]
        if not sub.empty and "text_path" in sub.columns:
            raw = str(sub.iloc[0]["text_path"]).strip()
            if raw:
                p = Path(raw.replace("\\", "/"))
                if p.exists():
                    return ("html" if p.name.endswith(".html.txt") else "txt", p)
    if key2txt_path.exists():
        k2 = pd.read_csv(key2txt_path, sep="\t", dtype=str).fillna("")
        sub2 = k2[k2["key"].astype(str).str.strip() == key]
        if not sub2.empty and "txt_path" in sub2.columns:
            raw2 = str(sub2.iloc[0]["txt_path"]).strip()
            if raw2:
                p2 = Path("data/cleaned/content_goren_2025") / raw2.replace("\\", "/")
                if p2.exists():
                    return ("html" if p2.name.endswith(".html.txt") else "txt", p2)
    return ("", None)


def extract_tables_from_source_text(source_text: str, source_used: str) -> list[TableBlock]:
    lines = source_text.splitlines()
    tables: list[TableBlock] = []

    marker_re = re.compile(r"^===\s*TABLE\s+(\d+)\s*\(TSV\)\s*===", re.IGNORECASE)
    starts: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        m = marker_re.match(line.strip())
        if m:
            starts.append((i, int(m.group(1))))

    for s_idx, (start_line, table_index) in enumerate(starts):
        end_line = starts[s_idx + 1][0] if s_idx + 1 < len(starts) else len(lines)
        block_lines = [ln for ln in lines[start_line + 1 : end_line] if ln.strip()]
        tsv_lines = [ln for ln in block_lines if "\t" in ln]
        if len(tsv_lines) < 2:
            continue
        rows_raw = [ln.split("\t") for ln in tsv_lines]
        max_len = max(len(r) for r in rows_raw)
        rows = [r + [""] * (max_len - len(r)) for r in rows_raw]

        # Some extracted tables contain a super-header row followed by the real header row.
        # Pick the row with maximum non-empty cells as the operational header.
        non_empty_counts = [sum(1 for c in r if str(c).strip()) for r in rows]
        header_idx = max(range(len(rows)), key=lambda i: (non_empty_counts[i], len(rows[i]), -i))
        header = [c.strip() for c in rows[header_idx]]
        body = [[c.strip() for c in r] for i, r in enumerate(rows) if i != header_idx]
        caption = ""
        for back in range(max(0, start_line - 6), start_line):
            if "table" in lines[back].lower():
                caption = lines[back].strip()
        anchor = f"{source_used}_table:{table_index}"
        tables.append(
            TableBlock(
                table_index=table_index,
                header=header,
                rows=body,
                caption=caption,
                source_anchor=anchor,
            )
        )

    return sorted(tables, key=lambda t: t.table_index)


def classify_codebook_table(
    tbl: TableBlock,
    *,
    max_code_abs: float,
    max_code_unique: int,
) -> bool:
    if not tbl.header or len(tbl.header) < 3 or not tbl.rows:
        return False
    header_vals = [str(c).strip() for c in tbl.header[1:]]
    header_is_discrete, _, _ = analyze_discrete_code_column(
        header_vals,
        max_code_abs=max_code_abs,
        max_unique=max_code_unique,
        require_repeated=False,
    )
    if not header_is_discrete:
        return False
    has_factor_rows = 0
    for row in tbl.rows:
        if not row:
            continue
        label = row[0]
        values = row[1:]
        n_num = sum(1 for v in values if parse_first_float(v) is not None)
        if label and n_num >= 2:
            has_factor_rows += 1
    return has_factor_rows >= 2


def classify_runs_table(
    tbl: TableBlock,
    *,
    max_code_abs: float,
    max_code_unique: int,
) -> bool:
    if len(tbl.header) < 5 or len(tbl.rows) < 4:
        return False
    col_values: list[list[str]] = [[] for _ in tbl.header]
    for row in tbl.rows:
        rr = row + [""] * (len(tbl.header) - len(row))
        for i, v in enumerate(rr[: len(tbl.header)]):
            if str(v).strip():
                col_values[i].append(str(v).strip())

    coded_cols = 0
    continuous_cols = 0
    for vals in col_values[1:]:
        if not vals:
            continue
        is_discrete_code, _, uniq = analyze_discrete_code_column(
            vals,
            max_code_abs=max_code_abs,
            max_unique=max_code_unique,
        )
        if is_discrete_code:
            coded_cols += 1
            continue
        num_ratio = sum(1 for v in vals if parse_first_float(v) is not None) / len(vals)
        continuous_min_unique = max(5, int(0.5 * len(vals)))
        if num_ratio >= 0.75 and (uniq > max_code_unique or uniq >= continuous_min_unique):
            continuous_cols += 1
    return coded_cols >= 2 and (continuous_cols >= 1 or len(tbl.rows) >= 8)


def match_factor_key(run_factor: str, codebook_keys: list[str]) -> str | None:
    rf = normalize_factor_name(run_factor)
    if rf in codebook_keys:
        return rf
    for cb in codebook_keys:
        if rf and (rf in cb or cb in rf):
            return cb
    rf_tokens = set(rf.split("_"))
    best = None
    best_score = 0
    for cb in codebook_keys:
        cb_tokens = set(cb.split("_"))
        score = len(rf_tokens & cb_tokens)
        if score > best_score:
            best = cb
            best_score = score
    return best if best_score >= 1 else None


def build_codebook(table: TableBlock) -> dict[str, dict[str, dict[str, Any]]]:
    code_objs = [parse_code_value(c) for c in table.header[1:]]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for row_idx, row in enumerate(table.rows, start=1):
        row = row + [""] * (len(table.header) - len(row))
        label_raw, default_unit = parse_factor_label(row[0])
        factor_key = normalize_factor_name(label_raw)
        if not factor_key:
            continue
        for i, (code_raw, code_num, code_canon) in enumerate(code_objs, start=1):
            if code_canon is None:
                continue
            cell = row[i] if i < len(row) else ""
            vnum = parse_first_float(cell)
            if vnum is None:
                continue
            out.setdefault(factor_key, {})
            out[factor_key][code_canon] = {
                "code_raw": code_raw,
                "code_num": code_num,
                "code_canon": code_canon,
                "value_num": vnum,
                "unit": default_unit or None,
                "value_raw": str(cell).strip(),
                "provenance_anchor": f"{table.source_anchor}:row{row_idx}",
                "caption": table.caption,
                "factor_name_original": label_raw,
                "factor_name_normalized": factor_key,
            }
    return out


def lookup_codebook_value(
    *,
    codebook_for_factor: dict[str, dict[str, Any]],
    run_code_canon: str | None,
    run_code_num: float | None,
    tol: float,
) -> tuple[dict[str, Any] | None, str]:
    if run_code_canon and run_code_canon in codebook_for_factor:
        return (codebook_for_factor[run_code_canon], "exact_code_canon")
    if run_code_num is None:
        return (None, "")
    for _, decoded in sorted(codebook_for_factor.items()):
        mapped_num = decoded.get("code_num")
        if mapped_num is None:
            continue
        if abs(float(mapped_num) - float(run_code_num)) <= tol:
            return (decoded, "numeric_tolerance")
    return (None, "")


def build_runs(
    table: TableBlock,
    *,
    max_code_abs: float,
    max_code_unique: int,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    header = table.header
    rows = table.rows
    if not header or not rows:
        return ([], [], [])

    col_vals: list[list[str]] = [[] for _ in header]
    for row in rows:
        rr = row + [""] * (len(header) - len(row))
        for i, v in enumerate(rr[: len(header)]):
            if str(v).strip():
                col_vals[i].append(str(v).strip())

    coded_cols: list[int] = []
    outcome_cols: list[int] = []
    for i in range(1, len(header)):
        vals = col_vals[i]
        if not vals:
            continue
        is_discrete_code, _, uniq = analyze_discrete_code_column(
            vals,
            max_code_abs=max_code_abs,
            max_unique=max_code_unique,
        )
        if is_discrete_code:
            coded_cols.append(i)
            continue
        num_ratio = sum(1 for v in vals if parse_first_float(v) is not None) / len(vals)
        continuous_min_unique = max(5, int(0.5 * len(vals)))
        if num_ratio >= 0.75 and (uniq > max_code_unique or uniq >= continuous_min_unique):
            outcome_cols.append(i)

    run_rows: list[dict[str, Any]] = []
    for r_idx, row in enumerate(rows, start=1):
        rr = row + [""] * (len(header) - len(row))
        first = str(rr[0]).strip()
        if not first:
            continue
        assignments: dict[str, dict[str, Any]] = {}
        for ci in coded_cols:
            code_raw, code_num, code_canon = parse_code_value(rr[ci])
            if code_canon is None:
                continue
            assignments[str(header[ci]).strip()] = {
                "code_raw": code_raw,
                "code_num": code_num,
                "code_canon": code_canon,
            }
        if not assignments:
            continue
        outcomes: dict[str, str] = {}
        for oi in outcome_cols:
            raw = str(rr[oi]).strip()
            if not raw:
                continue
            outcomes[str(header[oi]).strip()] = raw
        run_rows.append(
            {
                "run_label": first,
                "row_index": r_idx,
                "assignments": assignments,
                "outcomes": outcomes,
                "provenance_anchor": f"{table.source_anchor}:row{r_idx}",
            }
        )
    coded_names = [str(header[i]).strip() for i in coded_cols]
    outcome_names = [str(header[i]).strip() for i in outcome_cols]
    return (run_rows, coded_names, outcome_names)


def factor_core_copy_target(factor_key: str) -> str | None:
    allowlist = {
        "cpva": "surfactant_concentration",
        "cpva_mg_ml": "surfactant_concentration",
        "cplga": "polymer_mass_mg",
        "cplga_mg_ml": "polymer_mass_mg",
        "ph": "aqueous_phase_pH",
        "aqueous_phase_ph": "aqueous_phase_pH",
        "cpf": "drug_mass_mg",
        "cpf_mg_ml": "drug_mass_mg",
    }
    return allowlist.get(factor_key)


def derive_doe_coded_factors(
    *,
    run_id: str,
    extracted: pd.DataFrame,
    derived: pd.DataFrame,
    sample_manifest_path: Path,
    key2txt_path: Path,
    out_dir: Path,
    max_code_abs: float = DEFAULT_MAX_CODE_ABS,
    max_code_unique: int = DEFAULT_MAX_CODE_UNIQUE,
    code_match_tol: float = DEFAULT_CODE_MATCH_TOL,
    min_decoded_rate: float = DEFAULT_MIN_DECODED_RATE,
) -> dict[str, Any]:
    key_to_doi = infer_doi_map(extracted, sample_manifest_path)
    ex = extracted.copy()
    ex["key"] = ex["key"].astype(str).str.strip()
    ex["formulation_id"] = ex["formulation_id"].astype(str).str.strip()
    ex["group_key"] = ex["key"] + "::" + ex["formulation_id"]
    ex["doi"] = ex["key"].map(lambda k: key_to_doi.get(k, ""))
    ex["doi"] = ex["doi"].map(lambda d: d if d else "MISSING_DOI")

    diagnostics_rows: list[dict[str, Any]] = []
    doe_rows: list[dict[str, Any]] = []

    for (doi, key), key_df in ex.groupby(["doi", "key"], sort=True):
        source_used, source_path = discover_source_for_key(
            key=key,
            sample_manifest_path=sample_manifest_path,
            key2txt_path=key2txt_path,
        )
        fail_reasons: list[str] = []
        has_codebook_table = False
        has_runs_table = False
        coded_columns_count = 0
        decoded_rate = 0.0

        def emit_diag(status: str, fail_reason: str) -> None:
            hard_gate_failed = []
            if not has_codebook_table:
                hard_gate_failed.append("gate_missing_codebook_table")
            if not has_runs_table:
                hard_gate_failed.append("gate_missing_runs_table")
            if coded_columns_count < 2:
                hard_gate_failed.append("gate_coded_columns_lt_2")
            if decoded_rate < float(min_decoded_rate):
                hard_gate_failed.append("gate_decoded_rate_below_min")
            diagnostics_rows.append(
                {
                    "doi": doi,
                    "key": key,
                    "status": status,
                    "n_coded": 0,
                    "n_decoded": 0,
                    "n_factor_keys": 0,
                    "source_used": source_used,
                    "fail_reason": fail_reason,
                    "has_codebook_table": int(has_codebook_table),
                    "has_runs_table": int(has_runs_table),
                    "coded_columns_count": int(coded_columns_count),
                    "decoded_rate": float(decoded_rate),
                    "min_decoded_rate_required": float(min_decoded_rate),
                    "hard_gate_pass": int(len(hard_gate_failed) == 0),
                    "hard_gate_failed_reasons": "|".join(hard_gate_failed),
                }
            )

        if source_path is None:
            source_used = ""
            emit_diag("failed", "missing_source")
            continue

        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        tables = extract_tables_from_source_text(source_text=source_text, source_used=source_used)
        if not tables:
            emit_diag("failed", "no_tables_detected")
            continue

        codebook_tbls = [
            t
            for t in tables
            if classify_codebook_table(
                t,
                max_code_abs=max_code_abs,
                max_code_unique=max_code_unique,
            )
        ]
        runs_tbls = [
            t
            for t in tables
            if classify_runs_table(
                t,
                max_code_abs=max_code_abs,
                max_code_unique=max_code_unique,
            )
        ]
        has_codebook_table = len(codebook_tbls) > 0
        has_runs_table = len(runs_tbls) > 0

        if not runs_tbls:
            emit_diag("failed", "no_runs_table")
            continue

        if not codebook_tbls:
            fail_reasons.append("missing_codebook")

        codebook: dict[str, dict[str, dict[str, Any]]] = {}
        if codebook_tbls:
            codebook = build_codebook(codebook_tbls[0])

        runs, coded_names, _ = build_runs(
            runs_tbls[0],
            max_code_abs=max_code_abs,
            max_code_unique=max_code_unique,
        )
        coded_columns_count = len(coded_names)
        if not runs:
            emit_diag("failed", "no_runs_rows")
            continue

        by_formulation = {
            str(r["formulation_id"]).strip(): r for _, r in key_df.iterrows()
        }
        n_coded = 0
        n_decoded = 0
        factor_keys_seen: set[str] = set()

        for run in runs:
            run_label = str(run["run_label"]).strip()
            formulation_id = run_label if run_label in by_formulation else run_label.replace(" ", "")
            if formulation_id not in by_formulation:
                continue
            row_ref = by_formulation[formulation_id]
            group_key = f"{key}::{formulation_id}"
            trace_payload = {
                "group_key": group_key,
                "key": key,
                "formulation_id": formulation_id,
                "source_path": str(source_path),
                "provenance_anchor": run["provenance_anchor"],
            }
            trace_pointer = json.dumps(trace_payload, ensure_ascii=False, sort_keys=True)
            run_id_local = f"{key}::{runs_tbls[0].table_index}::{run['row_index']}"

            for factor_name, code_obj in sorted(run["assignments"].items()):
                factor_norm = normalize_factor_name(factor_name)
                factor_keys_seen.add(factor_norm)
                coded_value = str(code_obj.get("code_canon") or "")
                coded_raw = str(code_obj.get("code_raw") or "")
                coded_num = code_obj.get("code_num")
                coded_row = {
                    "run_id": run_id,
                    "group_key": group_key,
                    "key": key,
                    "formulation_id": formulation_id,
                    "field_name": f"doe_factor::{factor_norm}::coded",
                    "value": coded_value,
                    "rule_id": "R_DOE_CODED_FACTOR_EMIT_V1",
                    "derived_from": factor_name,
                    "value_source": "parsed_source_runs_table",
                    "trace_pointer": trace_pointer,
                    "doe_run_id": run_id_local,
                    "factor_name_original": factor_name,
                    "factor_name_normalized": factor_norm,
                    "factor_kind": "coded",
                    "factor_value_code": coded_value,
                    "factor_value_code_raw": coded_raw,
                    "factor_value_code_num": str(coded_num) if coded_num is not None else "",
                    "factor_value_code_canon": coded_value,
                    "factor_value_text": coded_value,
                    "factor_value_num": "",
                    "factor_unit": "",
                    "decoded_from": "",
                    "provenance_anchor": run["provenance_anchor"],
                }
                doe_rows.append(coded_row)
                n_coded += 1

                cb_key = match_factor_key(factor_name, list(codebook.keys())) if codebook else None
                decoded = None
                decode_match_method = ""
                if cb_key:
                    decoded, decode_match_method = lookup_codebook_value(
                        codebook_for_factor=codebook.get(cb_key, {}),
                        run_code_canon=coded_value,
                        run_code_num=coded_num,
                        tol=code_match_tol,
                    )
                if decoded is None:
                    fail_reasons.append("missing_level_code_mapping")
                    continue

                decoded_num = decoded.get("value_num")
                decoded_row = {
                    "run_id": run_id,
                    "group_key": group_key,
                    "key": key,
                    "formulation_id": formulation_id,
                    "field_name": f"doe_factor::{factor_norm}::decoded",
                    "value": str(decoded_num) if decoded_num is not None else str(decoded.get("value_raw", "")),
                    "rule_id": "R_DOE_CODED_FACTOR_DECODE_V1",
                    "derived_from": f"{factor_name}:{coded_value}",
                    "value_source": (
                        "decoded_from_codebook_numeric_tolerance"
                        if decode_match_method == "numeric_tolerance"
                        else "decoded_from_codebook"
                    ),
                    "trace_pointer": trace_pointer,
                    "doe_run_id": run_id_local,
                    "factor_name_original": factor_name,
                    "factor_name_normalized": factor_norm,
                    "factor_kind": "decoded",
                    "factor_value_code": coded_value,
                    "factor_value_code_raw": coded_raw,
                    "factor_value_code_num": str(coded_num) if coded_num is not None else "",
                    "factor_value_code_canon": coded_value,
                    "factor_value_text": str(decoded.get("value_raw", "")),
                    "factor_value_num": str(decoded_num) if decoded_num is not None else "",
                    "factor_unit": str(decoded.get("unit") or ""),
                    "decoded_from": coded_value,
                    "provenance_anchor": str(decoded.get("provenance_anchor", "")),
                }
                doe_rows.append(decoded_row)
                n_decoded += 1

        if n_coded > 0:
            decoded_rate = float(n_decoded) / float(n_coded)
        else:
            decoded_rate = 0.0

        hard_gate_failed: list[str] = []
        if not has_codebook_table:
            hard_gate_failed.append("gate_missing_codebook_table")
        if not has_runs_table:
            hard_gate_failed.append("gate_missing_runs_table")
        if coded_columns_count < 2:
            hard_gate_failed.append("gate_coded_columns_lt_2")
        if decoded_rate < float(min_decoded_rate):
            hard_gate_failed.append("gate_decoded_rate_below_min")
        hard_gate_pass = len(hard_gate_failed) == 0

        if n_decoded == 0 and "missing_codebook" in fail_reasons:
            status = "failed"
        elif n_decoded == n_coded and n_coded > 0:
            status = "ok"
        elif n_coded > 0:
            status = "partial"
        else:
            status = "failed"
            fail_reasons.append("no_mapped_runs_to_formulation_ids")
        if not hard_gate_pass:
            status = "failed"

        diagnostics_rows.append(
            {
                "doi": doi,
                "key": key,
                "status": status,
                "n_coded": int(n_coded),
                "n_decoded": int(n_decoded),
                "n_factor_keys": int(len(factor_keys_seen)),
                "source_used": source_used,
                "fail_reason": "|".join(sorted(set(fail_reasons))),
                "has_codebook_table": int(has_codebook_table),
                "has_runs_table": int(has_runs_table),
                "coded_columns_count": int(coded_columns_count),
                "decoded_rate": float(decoded_rate),
                "min_decoded_rate_required": float(min_decoded_rate),
                "hard_gate_pass": int(hard_gate_pass),
                "hard_gate_failed_reasons": "|".join(hard_gate_failed),
            }
        )

    gated_pairs = {
        (str(r.get("doi", "")), str(r.get("key", "")))
        for r in diagnostics_rows
        if int(r.get("hard_gate_pass", 0)) == 1 and str(r.get("status", "")).lower() in {"ok", "partial"}
    }
    if gated_pairs:
        keep_keys = {(k, d) for d, k in gated_pairs}
        doe_rows = [r for r in doe_rows if (str(r.get("key", "")), str(key_to_doi.get(str(r.get("key", "")), ""))) in keep_keys]
    else:
        doe_rows = []

    doe_rows = sorted(
        doe_rows,
        key=lambda r: (
            r["key"],
            r["formulation_id"],
            r["factor_name_normalized"],
            r["factor_kind"],
            r["factor_value_code"],
        ),
    )
    diagnostics_rows = sorted(diagnostics_rows, key=lambda r: (r["doi"], r["key"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    factor_cols = [
        "run_id",
        "group_key",
        "key",
        "formulation_id",
        "field_name",
        "value",
        "rule_id",
        "derived_from",
        "value_source",
        "trace_pointer",
        "doe_run_id",
        "factor_name_original",
        "factor_name_normalized",
        "factor_kind",
        "factor_value_code",
        "factor_value_code_raw",
        "factor_value_code_num",
        "factor_value_code_canon",
        "factor_value_text",
        "factor_value_num",
        "factor_unit",
        "decoded_from",
        "provenance_anchor",
    ]
    factor_df = pd.DataFrame(doe_rows, columns=factor_cols)
    factor_out = out_dir / "doe_factor_rows.tsv"
    factor_df.to_csv(factor_out, sep="\t", index=False)

    diag_cols = [
        "doi",
        "key",
        "status",
        "n_coded",
        "n_decoded",
        "n_factor_keys",
        "source_used",
        "fail_reason",
        "has_codebook_table",
        "has_runs_table",
        "coded_columns_count",
        "decoded_rate",
        "min_decoded_rate_required",
        "hard_gate_pass",
        "hard_gate_failed_reasons",
    ]
    diag_df = pd.DataFrame(diagnostics_rows, columns=diag_cols)
    diag_out = out_dir / "doe_decode_diagnostics.tsv"
    diag_df.to_csv(diag_out, sep="\t", index=False)

    derived_base = derived.copy()
    if derived_base.empty:
        derived_base = pd.DataFrame(
            columns=[
                "run_id",
                "group_key",
                "key",
                "formulation_id",
                "field_name",
                "value",
                "rule_id",
                "derived_from",
                "value_source",
                "trace_pointer",
            ]
        )
    derived_new = pd.concat([derived_base, factor_df[derived_base.columns]], ignore_index=True)

    existing = set(zip(derived_base["group_key"].astype(str), derived_base["field_name"].astype(str)))
    core_copy_rows: list[dict[str, str]] = []
    for _, row in factor_df[factor_df["factor_kind"] == "decoded"].iterrows():
        target = factor_core_copy_target(str(row["factor_name_normalized"]))
        if not target:
            continue
        gk = str(row["group_key"])
        if (gk, target) in existing:
            continue
        val = str(row["factor_value_num"] or row["factor_value_text"])
        if not val:
            continue
        core_copy_rows.append(
            {
                "run_id": str(row["run_id"]),
                "group_key": gk,
                "key": str(row["key"]),
                "formulation_id": str(row["formulation_id"]),
                "field_name": target,
                "value": val,
                "rule_id": "R_DOE_DECODE_CORE_COPY_V1",
                "derived_from": f"doe_factor:{row['factor_name_original']}",
                "value_source": "decoded_from_codebook",
                "trace_pointer": str(row["trace_pointer"]),
            }
        )
        existing.add((gk, target))
    if core_copy_rows:
        derived_new = pd.concat([derived_new, pd.DataFrame(core_copy_rows)], ignore_index=True)

    derived_new = derived_new.sort_values(
        ["group_key", "field_name", "rule_id", "value", "trace_pointer"]
    ).reset_index(drop=True)

    summary = {
        "doe_factor_rows": int(len(factor_df)),
        "doe_coded_rows": int((factor_df["factor_kind"] == "coded").sum()) if len(factor_df) else 0,
        "doe_decoded_rows": int((factor_df["factor_kind"] == "decoded").sum()) if len(factor_df) else 0,
        "doe_core_copy_rows": int(len(core_copy_rows)),
        "diagnostics_rows": int(len(diag_df)),
        "output_doe_factor_rows": str(factor_out),
        "output_doe_decode_diagnostics": str(diag_out),
    }
    return {
        "derived_df": derived_new,
        "factor_df": factor_df,
        "diagnostics_df": diag_df,
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    input_tsv = Path(args.input_tsv) if args.input_tsv else Path(f"data/results/{run_id}/weak_labels__gemini.tsv")
    derived_tsv = (
        Path(args.derived_tsv)
        if args.derived_tsv
        else Path(f"data/results/{run_id}/benchmark_goren_2025/derived_values.tsv")
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(f"data/results/{run_id}/benchmark_goren_2025/derivation_v1")
    )
    sample_manifest = Path(args.sample_manifest)
    key2txt = Path(args.key2txt)

    required = [input_tsv, derived_tsv]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input file(s): {missing}")

    if out_dir.exists() and not args.overwrite and (out_dir / "doe_decode_diagnostics.tsv").exists():
        raise FileExistsError(f"DOE outputs already exist at {out_dir}; use --overwrite")

    extracted = pd.read_csv(input_tsv, sep="\t", dtype=str).fillna("")
    derived = pd.read_csv(derived_tsv, sep="\t", dtype=str).fillna("")
    result = derive_doe_coded_factors(
        run_id=run_id,
        extracted=extracted,
        derived=derived,
        sample_manifest_path=sample_manifest,
        key2txt_path=key2txt,
        out_dir=out_dir,
        max_code_abs=args.max_code_abs,
        max_code_unique=args.max_code_unique,
        code_match_tol=args.code_match_tol,
        min_decoded_rate=args.min_decoded_rate,
    )
    result["derived_df"].to_csv(derived_tsv, sep="\t", index=False)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

