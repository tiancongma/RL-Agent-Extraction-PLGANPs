#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


TABLEFIRST_COMMIT = "8187537"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deterministic audit10 regression check with golden expected-source validation."
    )
    p.add_argument("--run-id", required=True)
    p.add_argument("--before-xlsx", default="")
    p.add_argument("--after-xlsx", default="")
    return p.parse_args()


def _s(v: Any) -> str:
    return str(v or "").strip()


def _short80(v: Any) -> str:
    return _s(v)[:80]


def _safe_read_xlsx(path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, dtype=str).fillna("")


def _contains_numeric_soft(text: str, raw_value: str) -> bool:
    t = _s(text).replace("\u2212", "-").lower()
    v = _s(raw_value).replace("\u2212", "-")
    if not t:
        return False
    if v and (v.lower() in t):
        return True
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?", v)
    if not m:
        return False
    try:
        target = float(m.group(0).replace(",", "."))
    except Exception:
        return False
    nums = []
    for x in re.findall(r"[-+]?\d+(?:[.,]\d+)?", t):
        try:
            nums.append(float(x.replace(",", ".")))
        except Exception:
            continue
    tol = max(0.5, abs(target) * 0.02)
    return any(abs(n - target) <= tol for n in nums)


def _values_equal_soft(a: str, b: str) -> bool:
    sa = _s(a)
    sb = _s(b)
    if sa == sb:
        return True
    ma = re.search(r"[-+]?\d+(?:[.,]\d+)?", sa)
    mb = re.search(r"[-+]?\d+(?:[.,]\d+)?", sb)
    if not (ma and mb):
        return False
    try:
        va = float(ma.group(0).replace(",", "."))
        vb = float(mb.group(0).replace(",", "."))
    except Exception:
        return False
    return abs(va - vb) <= max(0.01, abs(vb) * 0.005)


def discover_before_after(run_id: str, before_arg: str, after_arg: str) -> tuple[Path, Path]:
    audit_dir = Path("data/results") / run_id / "step1_dev" / "audit_pack"
    if not audit_dir.exists():
        raise FileNotFoundError(f"Audit directory not found: {audit_dir}")

    if after_arg:
        after = Path(after_arg)
    else:
        cands = sorted(audit_dir.glob("audit_pack__human_evidence_v1__tablefirst_v1.xlsx"))
        if len(cands) != 1:
            raise RuntimeError("Could not uniquely discover AFTER workbook. Provide --after-xlsx.")
        after = cands[0]
    if not after.exists():
        raise FileNotFoundError(f"AFTER workbook not found: {after}")

    if before_arg:
        before = Path(before_arg)
        if not before.exists():
            raise FileNotFoundError(f"BEFORE workbook not found: {before}")
        return before, after

    # No explicit BEFORE: print historical candidates and require explicit user choice if ambiguous.
    all_cands = sorted(Path("data/results").rglob("audit_pack__human_evidence_v1*.xlsx"))
    candidates = [p for p in all_cands if p.resolve() != after.resolve()]
    print(f"HISTORICAL_BEFORE_BASELINE_CANDIDATES (relative to table-first commit {TABLEFIRST_COMMIT}):")
    for p in candidates:
        print(str(p))
    if len(candidates) == 1:
        return candidates[0], after
    raise RuntimeError("Multiple BEFORE candidates found. Re-run with --before-xlsx <chosen_path>.")


def find_golden_workbook(step1_dir: Path) -> tuple[Path, str]:
    # Preferred 10-row workbook
    preferred = step1_dir / "audit_pack" / "dev_human_optimization_audit_10__numeric_mismatch_v1.xlsx"
    if preferred.exists():
        try:
            df = _safe_read_xlsx(preferred, "audit10")
            cols = set(df.columns)
            if {"reviewer_true_source", "reviewer_root_issue", "suspected_layer"}.issubset(cols):
                return preferred, "audit10"
        except Exception:
            pass

    # Fallback: any workbook with reviewer columns
    for p in sorted((step1_dir / "audit_pack").glob("*.xlsx")):
        try:
            xl = pd.ExcelFile(p)
        except Exception:
            continue
        for sh in xl.sheet_names:
            try:
                d = _safe_read_xlsx(p, sh)
            except Exception:
                continue
            cols = set(d.columns)
            if {"reviewer_true_source", "reviewer_root_issue", "suspected_layer"}.issubset(cols):
                return p, sh
    raise RuntimeError("No workbook found with reviewer_true_source/reviewer_root_issue/suspected_layer columns.")


def build_regression_and_golden(step1_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    wb_path, sheet = find_golden_workbook(step1_dir)
    src = _safe_read_xlsx(wb_path, sheet)
    if "qc_fail_type" in src.columns:
        src = src[src["qc_fail_type"].astype(str) == "numeric_token_mismatch"].copy()

    # Part 1 file
    reg_cols = {
        "zotero_key": "zotero_key",
        "field_name": "field_name",
        "extracted_value_raw": "extracted_value_raw",
        "extracted_value_canon": "extracted_value_canon",
        "evidence_span_start": "evidence_span_start",
        "evidence_span_end": "evidence_span_end",
        "table_csv_path_before": "table_csv_path",
        "evidence_pointer_raw_before": "evidence_pointer_raw",
        "qc_fail_type": "qc_fail_type",
    }
    reg_df = pd.DataFrame()
    for dst, src_col in reg_cols.items():
        reg_df[dst] = src[src_col].astype(str) if src_col in src.columns else ""
    reg_df = reg_df.sort_values(
        ["zotero_key", "field_name", "extracted_value_raw"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    reg_path = step1_dir / "audit10_regression_cases.tsv"
    reg_df.to_csv(reg_path, sep="\t", index=False)

    # Golden expected file
    golden_map = {
        "zotero_key": "zotero_key",
        "field_name": "field_name",
        "extracted_value_raw": "extracted_value_raw",
        "reviewer_true_source": "reviewer_true_source",
        "reviewer_root_issue": "reviewer_root_issue",
        "suspected_layer": "suspected_layer",
        "evidence_span_start": "evidence_span_start",
        "evidence_span_end": "evidence_span_end",
        "evidence_span_id": "evidence_span_id",
        "evidence_block_id": "evidence_block_id",
        "table_filename": "table_filename",
    }
    golden_df = pd.DataFrame()
    for dst, src_col in golden_map.items():
        golden_df[dst] = src[src_col].astype(str) if src_col in src.columns else ""
    golden_df = golden_df.sort_values(
        ["zotero_key", "field_name", "extracted_value_raw"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    golden_path = step1_dir / "audit10_golden_expected.tsv"
    golden_df.to_csv(golden_path, sep="\t", index=False)

    print(f"golden_source_workbook={wb_path}")
    print(f"golden_source_sheet={sheet}")
    print(f"total_cases_extracted={len(reg_df)}")
    print("cases_list:")
    for _, r in reg_df.iterrows():
        print(f"- ({_s(r['zotero_key'])}, {_s(r['field_name'])}, {_s(r['extracted_value_raw'])})")
    return reg_df, golden_df, reg_path, golden_path


def _field_value_col(field_name: str, df: pd.DataFrame) -> str:
    m = {
        "encapsulation_efficiency_percent": "EE_raw",
        "size_nm": "size_raw",
        "pdi": "pdi_raw",
    }
    c = m.get(_s(field_name), "")
    if c in df.columns:
        return c
    if "extracted_value_raw" in df.columns:
        return "extracted_value_raw"
    return ""


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in [
        "zotero_key",
        "evidence_span_id",
        "evidence_span_start",
        "evidence_span_end",
        "evidence_source_type",
        "evidence_pointer_raw",
        "table_csv_path",
        "table_cell_text",
        "table_row_text",
        "table_selection_status",
        "human_review_tag",
        "evidence_text",
    ]:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].astype(str)
    return out


def _deterministic_sort(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["_span_sort"] = pd.to_numeric(d.get("evidence_span_start", ""), errors="coerce").fillna(10**12)
    return d.sort_values(
        ["_span_sort", "evidence_pointer_raw", "table_csv_path"],
        ascending=[True, True, True],
        kind="mergesort",
    )


def _apply_field_filter_if_available(m: pd.DataFrame, field_name: str) -> pd.DataFrame:
    if "derived_field_name" not in m.columns:
        return m
    d = m["derived_field_name"].astype(str).str.strip()
    if not (d != "").any():
        return m
    m2 = m[d == _s(field_name)].copy()
    if not m2.empty:
        return m2
    return m


def match_case(df: pd.DataFrame, g: pd.Series) -> tuple[pd.Series | None, bool, str]:
    k = _s(g.get("zotero_key", ""))
    f = _s(g.get("field_name", ""))
    v = _s(g.get("extracted_value_raw", ""))
    sid = _s(g.get("evidence_span_id", ""))
    ss = _s(g.get("evidence_span_start", ""))
    se = _s(g.get("evidence_span_end", ""))

    base = df[df["zotero_key"] == k].copy()
    if base.empty:
        return None, False, "no_key_match"

    # 1) span_id
    if sid and "evidence_span_id" in base.columns:
        m = base[base["evidence_span_id"].astype(str) == sid].copy()
        if f:
            m = _apply_field_filter_if_available(m, f)
        val_col = _field_value_col(f, m)
        if v and val_col:
            m2 = m[m[val_col].astype(str).apply(lambda x: _values_equal_soft(x, v))]
            if m2.empty:
                m = m.iloc[0:0]
            else:
                m = m2
        if len(m) == 1:
            return m.iloc[0], False, "match_by_span_id"
        if len(m) > 1:
            return _deterministic_sort(m).iloc[0], False, "match_by_span_id_multi_deterministic"

    # 2) span start/end
    if ss and se and "evidence_span_start" in base.columns and "evidence_span_end" in base.columns:
        m = base[
            (base["evidence_span_start"].astype(str) == ss)
            & (base["evidence_span_end"].astype(str) == se)
        ].copy()
        if f:
            m = _apply_field_filter_if_available(m, f)
        val_col = _field_value_col(f, m)
        if v and val_col:
            m2 = m[m[val_col].astype(str).apply(lambda x: _values_equal_soft(x, v))]
            if m2.empty:
                m = m.iloc[0:0]
            else:
                m = m2
        if len(m) == 1:
            return m.iloc[0], False, "match_by_span_start_end"
        if len(m) > 1:
            return _deterministic_sort(m).iloc[0], False, "match_by_span_start_end_multi_deterministic"

    # 3) fallback by key+field+value (must be unique)
    m = base.copy()
    if f:
        m = _apply_field_filter_if_available(m, f)
    val_col = _field_value_col(f, m)
    if val_col:
        m = m[m[val_col].astype(str).apply(lambda x: _values_equal_soft(x, v))].copy()
    if len(m) == 1:
        return m.iloc[0], False, "match_by_key_field_value_unique"
    if len(m) > 1:
        return None, True, "ambiguous_key_field_value"
    return None, False, "no_match"


def improvement_status(
    before_src: str,
    after_src: str,
    before_table_nonempty: bool,
    after_table_nonempty: bool,
) -> str:
    if (not before_table_nonempty) and after_table_nonempty:
        return "improved_table_binding"
    if before_src == "text" and after_src == "table":
        return "source_corrected"
    if (not after_table_nonempty) and (after_src == before_src):
        return "unchanged"
    if (not after_table_nonempty) and before_table_nonempty:
        return "regressed"
    return "other_change"


def write_expected_summary(path: Path, exp_df: pd.DataFrame, ambiguous: list[str]) -> None:
    total = len(exp_df)
    pass_n = int((exp_df["expected_check_pass"] == True).sum()) if total else 0
    fail_n = total - pass_n
    failed = exp_df[exp_df["expected_check_pass"] == False][
        ["zotero_key", "field_name", "extracted_value_raw", "expected_fail_reason"]
    ]

    lines = [
        "# Audit10 Expected Source Summary (Table-First v1)",
        "",
        f"- total_cases: {total}",
        f"- n_expected_pass: {pass_n}",
        f"- n_expected_fail: {fail_n}",
        f"- n_ambiguous: {len(ambiguous)}",
        "",
        "- ambiguous_cases:",
    ]
    if ambiguous:
        for x in ambiguous:
            lines.append(f"  - {x}")
    else:
        lines.append("  - none")
    lines.append("")
    lines.append("- failed_expected_cases:")
    if failed.empty:
        lines.append("  - none")
    else:
        for _, r in failed.iterrows():
            lines.append(
                f"  - ({_s(r['zotero_key'])}, {_s(r['field_name'])}, {_s(r['extracted_value_raw'])}) reason={_s(r['expected_fail_reason'])}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_regression_summary(path: Path, comp_df: pd.DataFrame) -> None:
    counts = comp_df["improvement_status"].value_counts().to_dict() if len(comp_df) else {}
    reg = comp_df[comp_df["improvement_status"] == "regressed"][
        ["zotero_key", "field_name", "extracted_value_raw"]
    ]
    lines = [
        "# Audit10 Regression Summary (Table-First v1)",
        "",
        f"- total_cases: {len(comp_df)}",
        f"- n_improved_table_binding: {int(counts.get('improved_table_binding', 0))}",
        f"- n_source_corrected: {int(counts.get('source_corrected', 0))}",
        f"- n_unchanged: {int(counts.get('unchanged', 0))}",
        f"- n_regressed: {int(counts.get('regressed', 0))}",
        "",
        "- list_of_regressed_cases:",
    ]
    if reg.empty:
        lines.append("  - none")
    else:
        for _, r in reg.iterrows():
            lines.append(
                f"  - ({_s(r['zotero_key'])}, {_s(r['field_name'])}, {_s(r['extracted_value_raw'])})"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_id = _s(args.run_id)
    step1_dir = Path("data/results") / run_id / "step1_dev"
    if not step1_dir.exists():
        raise FileNotFoundError(f"Step1 folder not found: {step1_dir}")

    before_xlsx, after_xlsx = discover_before_after(run_id, args.before_xlsx, args.after_xlsx)
    reg_df, golden_df, reg_path, golden_path = build_regression_and_golden(step1_dir)

    before_df = _prepare_df(_safe_read_xlsx(before_xlsx, "audit_cases"))
    after_df = _prepare_df(_safe_read_xlsx(after_xlsx, "audit_cases"))

    comp_rows: list[dict[str, Any]] = []
    expected_rows: list[dict[str, Any]] = []
    ambiguous_cases: list[str] = []

    for _, g in golden_df.iterrows():
        key = _s(g.get("zotero_key", ""))
        field = _s(g.get("field_name", ""))
        value = _s(g.get("extracted_value_raw", ""))
        rid = f"{key}|{field}|{value}"

        b, b_amb, b_reason = match_case(before_df, g)
        a, a_amb, a_reason = match_case(after_df, g)
        if b_amb or a_amb:
            ambiguous_cases.append(rid)

        before_src = _s(b.get("evidence_source_type", "")) if b is not None else ""
        after_src = _s(a.get("evidence_source_type", "")) if a is not None else ""
        before_table_path = _s(b.get("table_csv_path", "")) if b is not None else ""
        after_table_path = _s(a.get("table_csv_path", "")) if a is not None else ""
        before_table_nonempty = before_table_path != ""
        after_table_nonempty = after_table_path != ""
        before_pointer = _s(b.get("evidence_pointer_raw", "")) if b is not None else ""
        after_pointer = _s(a.get("evidence_pointer_raw", "")) if a is not None else ""

        comp_rows.append(
            {
                "zotero_key": key,
                "field_name": field,
                "extracted_value_raw": value,
                "before_evidence_source_type": before_src,
                "before_table_csv_path_nonempty": before_table_nonempty,
                "before_pointer_short": _short80(before_pointer),
                "before_table_selection_status": _s(b.get("table_selection_status", "") if b is not None else ""),
                "after_evidence_source_type": after_src,
                "after_table_csv_path_nonempty": after_table_nonempty,
                "after_pointer_short": _short80(after_pointer),
                "after_table_selection_status": _s(a.get("table_selection_status", "") if a is not None else ""),
                "improvement_status": improvement_status(
                    before_src=before_src,
                    after_src=after_src,
                    before_table_nonempty=before_table_nonempty,
                    after_table_nonempty=after_table_nonempty,
                ),
            }
        )

        reviewer_true_source = _s(g.get("reviewer_true_source", "")).lower()
        exp_pass = True
        exp_reason = "pass"
        if reviewer_true_source == "table":
            table_cell_nonempty = _s(a.get("table_cell_text", "") if a is not None else "") != ""
            type_ok = (after_src == "table") or after_pointer.startswith("table|")
            bind_ok = after_table_nonempty or table_cell_nonempty
            if not (bind_ok and type_ok):
                exp_pass = False
                exp_reason = "FAILED_expected_table_binding"
        elif reviewer_true_source == "text":
            text_blob = _s(a.get("evidence_text", "") if a is not None else "")
            pointer_l = after_pointer.lower()
            has_numeric_pointer = "numeric_locate" in pointer_l
            if not (_contains_numeric_soft(text_blob, value) or has_numeric_pointer):
                exp_pass = False
                exp_reason = "FAILED_expected_text_anchor"
        else:
            exp_pass = False
            exp_reason = "FAILED_missing_reviewer_true_source"

        expected_rows.append(
            {
                "zotero_key": key,
                "field_name": field,
                "extracted_value_raw": value,
                "reviewer_true_source": _s(g.get("reviewer_true_source", "")),
                "reviewer_root_issue": _s(g.get("reviewer_root_issue", "")),
                "suspected_layer": _s(g.get("suspected_layer", "")),
                "before_match_reason": b_reason,
                "after_match_reason": a_reason,
                "after_evidence_source_type": after_src,
                "after_table_csv_path_nonempty": after_table_nonempty,
                "after_table_cell_text_nonempty": _s(a.get("table_cell_text", "") if a is not None else "") != "",
                "after_evidence_pointer_raw": _short80(after_pointer),
                "expected_check_pass": exp_pass,
                "expected_fail_reason": exp_reason,
            }
        )

    comp_df = pd.DataFrame(comp_rows)
    expected_df = pd.DataFrame(expected_rows)

    comp_path = step1_dir / "audit10_regression_comparison__tablefirst_v1.tsv"
    reg_sum_path = step1_dir / "audit10_regression_summary__tablefirst_v1.md"
    exp_path = step1_dir / "audit10_expected_check__tablefirst_v1.tsv"
    exp_sum_path = step1_dir / "audit10_expected_summary__tablefirst_v1.md"

    comp_df.to_csv(comp_path, sep="\t", index=False)
    expected_df.to_csv(exp_path, sep="\t", index=False)
    write_regression_summary(reg_sum_path, comp_df)
    write_expected_summary(exp_sum_path, expected_df, ambiguous_cases)

    # Console summary
    counts = comp_df["improvement_status"].value_counts().to_dict() if len(comp_df) else {}
    print(f"total_cases={len(comp_df)}")
    print(f"n_improved_table_binding={int(counts.get('improved_table_binding', 0))}")
    print(f"n_source_corrected={int(counts.get('source_corrected', 0))}")
    print(f"n_unchanged={int(counts.get('unchanged', 0))}")
    print(f"n_regressed={int(counts.get('regressed', 0))}")
    exp_fail = int((expected_df["expected_check_pass"] == False).sum()) if len(expected_df) else 0
    print(f"n_expected_fail={exp_fail}")
    print(f"n_ambiguous={len(ambiguous_cases)}")

    failed_expected = expected_df[expected_df["expected_check_pass"] == False]
    if len(ambiguous_cases) > 0 or not failed_expected.empty:
        print("REGRESSION DETECTED")
        if ambiguous_cases:
            print("ambiguous_cases:")
            for x in ambiguous_cases:
                print(f"- {x}")
        if not failed_expected.empty:
            print("failed_expected_cases:")
            for _, r in failed_expected.iterrows():
                print(
                    f"- ({_s(r['zotero_key'])}, {_s(r['field_name'])}, {_s(r['extracted_value_raw'])}) reason={_s(r['expected_fail_reason'])}"
                )
        print(f"audit10_regression_cases_tsv={reg_path}")
        print(f"audit10_golden_expected_tsv={golden_path}")
        print(f"audit10_regression_comparison_tsv={comp_path}")
        print(f"audit10_expected_check_tsv={exp_path}")
        print(f"audit10_expected_summary_md={exp_sum_path}")
        print(f"run_id={run_id}")
        return 2

    print("Table-first binding regression test PASSED")
    print(f"audit10_regression_cases_tsv={reg_path}")
    print(f"audit10_golden_expected_tsv={golden_path}")
    print(f"audit10_regression_comparison_tsv={comp_path}")
    print(f"audit10_expected_check_tsv={exp_path}")
    print(f"audit10_expected_summary_md={exp_sum_path}")
    print(f"run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
