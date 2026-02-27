#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deterministic regression comparison for 10 manually reviewed numeric mismatch cases."
    )
    p.add_argument("--run-id", required=True)
    p.add_argument("--before-xlsx", default="")
    p.add_argument("--after-xlsx", default="")
    return p.parse_args()


def _s(v: Any) -> str:
    return str(v or "").strip()


def _bool_nonempty(v: Any) -> bool:
    return _s(v) != ""


def _short80(v: Any) -> str:
    return _s(v)[:80]


def _safe_read_xlsx(path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, dtype=str).fillna("")


def discover_xlsx(run_id: str, provided_before: str, provided_after: str) -> tuple[Path, Path]:
    audit_dir = Path("data/results") / run_id / "step1_dev" / "audit_pack"
    if not audit_dir.exists():
        raise FileNotFoundError(f"Audit directory not found: {audit_dir}")

    if provided_before:
        before = Path(provided_before)
    else:
        cands = sorted(audit_dir.glob("audit_pack__human_evidence_v1__static_rebuild.xlsx"))
        if len(cands) != 1:
            print("AMBIGUOUS_BEFORE_XLSX_CANDIDATES")
            for p in sorted(audit_dir.glob("audit_pack__human_evidence_v1__*.xlsx")):
                print(str(p))
            raise RuntimeError("Could not uniquely discover BEFORE workbook.")
        before = cands[0]

    if provided_after:
        after = Path(provided_after)
    else:
        cands = sorted(audit_dir.glob("audit_pack__human_evidence_v1__tablefirst_v1.xlsx"))
        if len(cands) != 1:
            print("AMBIGUOUS_AFTER_XLSX_CANDIDATES")
            for p in sorted(audit_dir.glob("audit_pack__human_evidence_v1__*.xlsx")):
                print(str(p))
            raise RuntimeError("Could not uniquely discover AFTER workbook.")
        after = cands[0]

    if not before.exists():
        raise FileNotFoundError(f"BEFORE workbook not found: {before}")
    if not after.exists():
        raise FileNotFoundError(f"AFTER workbook not found: {after}")
    return before, after


def build_regression_cases(run_id: str) -> tuple[pd.DataFrame, Path]:
    step1_dir = Path("data/results") / run_id / "step1_dev"
    audit_dir = step1_dir / "audit_pack"
    out_path = step1_dir / "audit10_regression_cases.tsv"
    src_10 = audit_dir / "dev_human_optimization_audit_10__numeric_mismatch_v1.xlsx"

    if src_10.exists():
        df = _safe_read_xlsx(src_10, "audit10")
        source_name = str(src_10)
    else:
        # Fallback path per requirement: use manually reviewed audit where reviewer_root_issue exists.
        fallback = sorted(audit_dir.glob("*.xlsx"))
        source_name = ""
        df = pd.DataFrame()
        for p in fallback:
            try:
                for sh in pd.ExcelFile(p).sheet_names:
                    d = _safe_read_xlsx(p, sh)
                    if "reviewer_root_issue" in d.columns and "qc_fail_type" in d.columns:
                        m = d[
                            d["reviewer_root_issue"].astype(str).str.strip().ne("")
                            & d["qc_fail_type"].astype(str).eq("numeric_token_mismatch")
                        ].copy()
                        if not m.empty:
                            df = m
                            source_name = str(p)
                            break
                if not df.empty:
                    break
            except Exception:
                continue
        if df.empty:
            raise RuntimeError(
                "Could not find dev_human_optimization_audit_10 workbook or a fallback reviewed workbook with reviewer_root_issue + numeric_token_mismatch."
            )

    cols = {
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
    out = pd.DataFrame()
    for dest, src in cols.items():
        out[dest] = df[src].astype(str) if src in df.columns else ""

    out = out[out["qc_fail_type"].astype(str).eq("numeric_token_mismatch")].copy()
    out = out.sort_values(
        ["zotero_key", "field_name", "extracted_value_raw"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out.to_csv(out_path, sep="\t", index=False)

    print(f"regression_cases_source={source_name}")
    print(f"total_cases_extracted={len(out)}")
    print("cases_list:")
    for _, r in out.iterrows():
        print(f"- ({_s(r['zotero_key'])}, {_s(r['field_name'])}, {_s(r['extracted_value_raw'])})")
    return out, out_path


def normalized_field_value_col(field_name: str, df: pd.DataFrame) -> str:
    mapping = {
        "encapsulation_efficiency_percent": "EE_raw",
        "size_nm": "size_raw",
        "pdi": "pdi_raw",
    }
    col = mapping.get(_s(field_name), "")
    if col and col in df.columns:
        return col
    if "extracted_value_raw" in df.columns:
        return "extracted_value_raw"
    return ""


def pick_match(df: pd.DataFrame, zotero_key: str, field_name: str, extracted_value_raw: str) -> pd.Series | None:
    if df.empty:
        return None
    d = df.copy()
    d["zotero_key"] = d.get("zotero_key", "").astype(str)
    d = d[d["zotero_key"] == _s(zotero_key)].copy()
    if d.empty:
        return None

    val_col = normalized_field_value_col(field_name, d)
    if val_col:
        d = d[d[val_col].astype(str) == _s(extracted_value_raw)].copy()
    if d.empty:
        return None

    span_col = "evidence_span_start" if "evidence_span_start" in d.columns else ""
    if span_col:
        d["_span_sort"] = pd.to_numeric(d[span_col], errors="coerce").fillna(10**12)
        d = d.sort_values(
            ["_span_sort", "evidence_pointer_raw", "table_csv_path"],
            ascending=[True, True, True],
            kind="mergesort",
        )
    else:
        d = d.sort_values(["evidence_pointer_raw", "table_csv_path"], ascending=[True, True], kind="mergesort")
    return d.iloc[0]


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


def write_summary(path: Path, comp: pd.DataFrame) -> None:
    total = int(len(comp))
    counts = comp["improvement_status"].value_counts().to_dict() if total else {}
    reg = comp[comp["improvement_status"] == "regressed"][
        ["zotero_key", "field_name", "extracted_value_raw"]
    ].copy()

    lines: list[str] = [
        "# Audit10 Regression Summary (Table-First v1)",
        "",
        f"- total_cases: {total}",
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
    before_xlsx, after_xlsx = discover_xlsx(run_id, args.before_xlsx, args.after_xlsx)
    cases_df, cases_path = build_regression_cases(run_id)

    before_df = _safe_read_xlsx(before_xlsx, "audit_cases")
    after_df = _safe_read_xlsx(after_xlsx, "audit_cases")
    for d in [before_df, after_df]:
        for c in ["zotero_key", "derived_field_name", "EE_raw", "size_raw", "pdi_raw"]:
            if c not in d.columns:
                d[c] = ""
            d[c] = d[c].astype(str)

    comp_rows: list[dict[str, Any]] = []
    for _, case in cases_df.iterrows():
        k = _s(case.get("zotero_key", ""))
        f = _s(case.get("field_name", ""))
        v = _s(case.get("extracted_value_raw", ""))

        b = pick_match(before_df, k, f, v)
        a = pick_match(after_df, k, f, v)

        before_src = _s(b.get("evidence_source_type", "")) if b is not None else ""
        after_src = _s(a.get("evidence_source_type", "")) if a is not None else ""
        before_path = _s(b.get("table_csv_path", "")) if b is not None else ""
        after_path = _s(a.get("table_csv_path", "")) if a is not None else ""

        before_nonempty = _bool_nonempty(before_path)
        after_nonempty = _bool_nonempty(after_path)

        comp_rows.append(
            {
                "zotero_key": k,
                "field_name": f,
                "extracted_value_raw": v,
                "before_evidence_source_type": before_src,
                "before_table_csv_path_nonempty": before_nonempty,
                "before_pointer_short": _short80(b.get("evidence_pointer_raw", "") if b is not None else ""),
                "before_table_selection_status": _s(b.get("table_selection_status", "") if b is not None else ""),
                "after_evidence_source_type": after_src,
                "after_table_csv_path_nonempty": after_nonempty,
                "after_pointer_short": _short80(a.get("evidence_pointer_raw", "") if a is not None else ""),
                "after_table_selection_status": _s(a.get("table_selection_status", "") if a is not None else ""),
                "improvement_status": improvement_status(
                    before_src=before_src,
                    after_src=after_src,
                    before_table_nonempty=before_nonempty,
                    after_table_nonempty=after_nonempty,
                ),
            }
        )

    comp_df = pd.DataFrame(comp_rows)
    step1_dir = Path("data/results") / run_id / "step1_dev"
    comp_path = step1_dir / "audit10_regression_comparison__tablefirst_v1.tsv"
    sum_path = step1_dir / "audit10_regression_summary__tablefirst_v1.md"

    comp_df.to_csv(comp_path, sep="\t", index=False)
    write_summary(sum_path, comp_df)

    counts = comp_df["improvement_status"].value_counts().to_dict() if len(comp_df) else {}
    print(f"total_cases={len(comp_df)}")
    print(f"n_improved_table_binding={int(counts.get('improved_table_binding', 0))}")
    print(f"n_source_corrected={int(counts.get('source_corrected', 0))}")
    print(f"n_unchanged={int(counts.get('unchanged', 0))}")
    print(f"n_regressed={int(counts.get('regressed', 0))}")

    reg = comp_df[comp_df["improvement_status"] == "regressed"]
    if not reg.empty:
        keys = sorted(set(reg["zotero_key"].astype(str).tolist()))
        print("REGRESSION DETECTED")
        print("affected_keys=" + ",".join(keys))
        print(f"audit10_regression_cases_tsv={cases_path}")
        print(f"audit10_regression_comparison_tsv={comp_path}")
        print(f"audit10_regression_summary_md={sum_path}")
        print(f"run_id={run_id}")
        return 2

    print("Table-first binding regression test PASSED")
    print(f"audit10_regression_cases_tsv={cases_path}")
    print(f"audit10_regression_comparison_tsv={comp_path}")
    print(f"audit10_regression_summary_md={sum_path}")
    print(f"run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
