#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from derive_doe_coded_factors_v1 import derive_doe_coded_factors


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"
DEFAULT_RULE_REGISTRY = "data/benchmark/goren_2025/rules/derivation_rule_registry.v1.json"

FIELD_ALIASES = {
    "polymer_mw_kDa": ["polymer_mw_kDa", "plga_mw_kDa"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legacy or branch-only modeling helper: derive normalized formulation fields from extraction output without changing extraction logic."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--input-tsv",
        default="",
        help="Defaults to data/results/<run_id>/weak_labels__gemini.tsv",
    )
    parser.add_argument(
        "--rule-registry",
        default=DEFAULT_RULE_REGISTRY,
        help="Immutable derivation rule registry JSON.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Defaults to data/results/<run_id>/benchmark_goren_2025",
    )
    return parser.parse_args()


def parse_float(raw: Any) -> float | None:
    if raw is None or pd.isna(raw):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().replace(",", "")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_mass_to_mg(raw: Any) -> float | None:
    if raw is None or pd.isna(raw):
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    text = text.replace("μ", "u")
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(mg|g|ug|mcg|ng)\b", text)
    if not match:
        # Fallback for numeric text with unknown unit; treated as mg to avoid silent drop.
        return parse_float(text)
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "g":
        return value * 1000.0
    if unit in {"ug", "mcg"}:
        return value / 1000.0
    if unit == "ng":
        return value / 1_000_000.0
    return value


def parse_la_ga_ratio(raw: Any) -> tuple[float | None, float | None, float | None]:
    if raw is None or pd.isna(raw):
        return (None, None, None)
    text = str(raw).strip()
    if not text:
        return (None, None, None)
    text = text.replace(" ", "")
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            try:
                la = float(parts[0])
                ga = float(parts[1])
            except ValueError:
                return (None, None, None)
            total = la + ga
            if total == 0:
                return (None, None, None)
            la_fraction = la / total
            ga_fraction = ga / total
            la_over_ga = (la / ga) if ga != 0 else None
            return (la_fraction, ga_fraction, la_over_ga)
    # If only one numeric value exists, preserve it as LA/GA ratio.
    scalar = parse_float(text)
    return (None, None, scalar)


def parse_mw_range(raw: Any) -> tuple[float | None, float | None]:
    if raw is None or pd.isna(raw):
        return (None, None)
    text = str(raw).strip().replace(",", "")
    if not text:
        return (None, None)
    vals = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not vals:
        return (None, None)
    if len(vals) == 1:
        v = float(vals[0])
        return (v, v)
    lo = float(vals[0])
    hi = float(vals[1])
    if lo <= hi:
        return (lo, hi)
    return (hi, lo)


def get_field_value(row: pd.Series, field_name: str) -> Any:
    for candidate in FIELD_ALIASES.get(field_name, [field_name]):
        if candidate in row.index:
            value = row.get(candidate, "")
            if str(value).strip():
                return value
    return row.get(field_name, "")


def parse_aqueous_organic_from_span(raw: Any) -> tuple[float | None, float | None]:
    if raw is None or pd.isna(raw):
        return (None, None)
    text = str(raw)
    if not text.strip():
        return (None, None)
    compact = re.sub(r"\s+", "", text.lower())

    # W1/O patterns.
    m_w1o = re.search(r"w1/o[:=]?(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)", compact)
    w1_o = None
    if m_w1o:
        a = float(m_w1o.group(1))
        b = float(m_w1o.group(2))
        if b != 0:
            w1_o = a / b

    # (W1+W2)/O patterns.
    m_wtot_o = re.search(r"\(w1\+w2\)/o[:=]?(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)", compact)
    wtot_o = None
    if m_wtot_o:
        a = float(m_wtot_o.group(1))
        b = float(m_wtot_o.group(2))
        if b != 0:
            wtot_o = a / b

    return (w1_o, wtot_o)


def make_trace_pointer(row: pd.Series, row_index: int) -> str:
    payload = {
        "row_index": int(row_index),
        "key": str(row.get("key", "") or ""),
        "formulation_id": str(row.get("formulation_id", "") or ""),
        "evidence_section": str(row.get("evidence_section", "") or ""),
        "evidence_span_start": str(row.get("evidence_span_start", "") or ""),
        "evidence_span_end": str(row.get("evidence_span_end", "") or ""),
    }
    return json.dumps(payload, ensure_ascii=False)


def is_paper_local_table_path(table_csv_path: Any, key: str) -> bool:
    p = str(table_csv_path or "").strip()
    k = str(key or "").strip()
    if not p or not k:
        return False
    parts = [str(x).lower() for x in Path(p).resolve().parts]
    if "tables" not in parts:
        return False
    idx = parts.index("tables")
    if idx + 1 >= len(parts):
        return False
    return parts[idx + 1] == k.lower()


def parse_row_text_pairs(table_row_text: Any) -> dict[str, str]:
    text = str(table_row_text or "").strip()
    out: dict[str, str] = {}
    if not text:
        return out
    for part in text.split("|"):
        seg = str(part).strip()
        if not seg or ":" not in seg:
            continue
        k, v = seg.split(":", 1)
        key = re.sub(r"\s+", " ", k).strip().lower()
        val = re.sub(r"\s+", " ", v).strip()
        if key:
            out[key] = val
    return out


def is_before_freeze_header(col_name: Any) -> bool:
    c = re.sub(r"\s+", " ", str(col_name or "").lower()).strip()
    return bool(
        re.search(r"\bbefore\b.*\bfreeze[- ]?dry(?:ing)?\b", c)
        or re.search(r"\bfreeze[- ]?dry(?:ing)?\b.*\bbefore\b", c)
    )


def is_after_freeze_header(col_name: Any) -> bool:
    c = re.sub(r"\s+", " ", str(col_name or "").lower()).strip()
    return bool(
        re.search(r"\bafter\b.*\bfreeze[- ]?dry(?:ing)?\b", c)
        or re.search(r"\bfreeze[- ]?dry(?:ing)?\b.*\bafter\b", c)
    )


def find_target_row_index(df: pd.DataFrame, row: pd.Series) -> int | None:
    row_index_raw = str(row.get("row_index", "")).strip()
    if row_index_raw.isdigit():
        idx = int(row_index_raw)
        if 0 <= idx < len(df):
            return idx

    pointer = str(row.get("evidence_pointer_raw", "") or row.get("evidence_ref", "")).strip()
    m = re.search(r"(?:row(?:_index)?|r)\s*[:=]\s*(\d+)", pointer, flags=re.IGNORECASE)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(df):
            return idx

    pairs = parse_row_text_pairs(row.get("table_row_text", ""))
    if pairs:
        best_idx = None
        best_score = -1
        for ridx in range(len(df)):
            rr = df.iloc[ridx]
            score = 0
            for k, v in pairs.items():
                if k not in [str(c).strip().lower() for c in df.columns]:
                    continue
                for c in df.columns:
                    ck = str(c).strip().lower()
                    if ck != k:
                        continue
                    cv = re.sub(r"\s+", " ", str(rr.get(c, "")).strip())
                    if cv == v:
                        score += 2
                    elif v and v in cv:
                        score += 1
            if score > best_score:
                best_score = score
                best_idx = ridx
        if best_idx is not None and best_score > 0:
            return int(best_idx)
    return None


def derive_baseline_size_before_freeze_drying(row: pd.Series) -> tuple[float | None, str, str]:
    key = str(row.get("key", "")).strip()
    table_csv_path = str(row.get("table_csv_path", "")).strip()
    if not is_paper_local_table_path(table_csv_path, key):
        return (None, "", "")
    table_path = Path(table_csv_path)
    if not table_path.exists():
        return (None, "", "")

    try:
        tdf = pd.read_csv(table_path, dtype=str).fillna("")
    except Exception:
        return (None, "", "")
    if tdf.empty:
        return (None, "", "")

    before_cols = [str(c) for c in tdf.columns if is_before_freeze_header(c)]
    after_cols = [str(c) for c in tdf.columns if is_after_freeze_header(c)]
    if not before_cols or not after_cols:
        return (None, "", "")

    row_idx = find_target_row_index(tdf, row)
    if row_idx is None:
        valid_rows = []
        for ridx in range(len(tdf)):
            rr = tdf.iloc[ridx]
            has_before = any(parse_float(rr.get(c, "")) is not None for c in before_cols)
            has_after = any(parse_float(rr.get(c, "")) is not None for c in after_cols)
            if has_before and has_after:
                valid_rows.append(ridx)
        if len(valid_rows) != 1:
            return (None, "", "")
        row_idx = int(valid_rows[0])

    rr = tdf.iloc[row_idx]
    before_col = ""
    before_val = None
    for c in before_cols:
        v = parse_float(rr.get(c, ""))
        if v is not None:
            before_col = c
            before_val = v
            break
    after_col = ""
    after_val = None
    for c in after_cols:
        v = parse_float(rr.get(c, ""))
        if v is not None:
            after_col = c
            after_val = v
            break

    if before_val is None or after_val is None:
        return (None, "", "")

    trace_payload = {
        "table_csv_path": str(table_path),
        "table_row_index": int(row_idx),
        "before_col": before_col,
        "before_cell_text": str(rr.get(before_col, "")),
        "after_col": after_col,
        "after_cell_text": str(rr.get(after_col, "")),
    }
    return (float(before_val), json.dumps(trace_payload, ensure_ascii=False), "table_cell_before_freeze_drying")


def add_value(
    out_rows: list[dict[str, Any]],
    *,
    run_id: str,
    group_key: str,
    key: str,
    formulation_id: str,
    field_name: str,
    value: Any,
    rule_id: str,
    derived_from: str,
    value_source: str,
    trace_pointer: str,
) -> None:
    if value is None:
        return
    txt = str(value).strip()
    if txt == "":
        return
    out_rows.append(
        {
            "run_id": run_id,
            "group_key": group_key,
            "key": key,
            "formulation_id": formulation_id,
            "field_name": field_name,
            "value": txt,
            "rule_id": rule_id,
            "derived_from": derived_from,
            "value_source": value_source,
            "trace_pointer": trace_pointer,
        }
    )


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    input_tsv = Path(args.input_tsv) if args.input_tsv else Path(f"data/results/{run_id}/weak_labels__gemini.tsv")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/results/{run_id}/benchmark_goren_2025")
    rule_registry = Path(args.rule_registry)

    if not input_tsv.exists():
        raise FileNotFoundError(f"Input extraction TSV not found: {input_tsv}")
    if not rule_registry.exists():
        raise FileNotFoundError(f"Derivation rule registry not found: {rule_registry}")

    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = pd.read_csv(input_tsv, sep="\t", dtype=str).fillna("")
    required_cols = ["key", "formulation_id"]
    missing = [c for c in required_cols if c not in extracted.columns]
    if missing:
        raise RuntimeError(f"Extraction TSV missing required columns: {missing}")

    derived_rows: list[dict[str, Any]] = []
    anchors_for_ao = {"plga_mass_mg", "drug_feed_amount_text", "la_ga_ratio", "polymer_mw_kDa"}

    for row_index, row in extracted.iterrows():
        key = str(row.get("key", "")).strip()
        formulation_id = str(row.get("formulation_id", "")).strip()
        group_key = f"{key}::{formulation_id}"
        trace_ptr = make_trace_pointer(row, row_index)

        # Direct parsed anchors.
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="small_molecule_name",
            value=row.get("drug_name", ""),
            rule_id="R_DIRECT_DRUG_NAME",
            derived_from="drug_name",
            value_source="extracted_anchor",
            trace_pointer=trace_ptr,
        )
        baseline_size, baseline_trace, baseline_source = derive_baseline_size_before_freeze_drying(row)
        if baseline_size is not None:
            add_value(
                derived_rows,
                run_id=run_id,
                group_key=group_key,
                key=key,
                formulation_id=formulation_id,
                field_name="size_nm__baseline_before_freeze_drying",
                value=baseline_size,
                rule_id="baseline_size_before_freeze_drying_v1",
                derived_from="table_before_after_freeze_drying_columns",
                value_source=baseline_source,
                trace_pointer=baseline_trace,
            )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="solvent",
            value=row.get("organic_solvent", ""),
            rule_id="R_DIRECT_ORGANIC_SOLVENT",
            derived_from="organic_solvent",
            value_source="extracted_anchor",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="surfactant_concentration",
            value=row.get("pva_conc_percent", ""),
            rule_id="R_DIRECT_SURFACTANT_CONC",
            derived_from="pva_conc_percent",
            value_source="extracted_anchor",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="particle_size",
            value=row.get("size_nm", ""),
            rule_id="R_DIRECT_PARTICLE_SIZE",
            derived_from="size_nm",
            value_source="extracted_anchor",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="EE",
            value=row.get("encapsulation_efficiency_percent", ""),
            rule_id="R_DIRECT_EE",
            derived_from="encapsulation_efficiency_percent",
            value_source="extracted_anchor",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="LC",
            value=row.get("loading_content_percent", ""),
            rule_id="R_DIRECT_LC",
            derived_from="loading_content_percent",
            value_source="extracted_anchor",
            trace_pointer=trace_ptr,
        )

        la_fraction, ga_fraction, la_over_ga = parse_la_ga_ratio(row.get("la_ga_ratio", ""))
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="la_fraction",
            value=la_fraction,
            rule_id="R_LAGA_PARSE_FRACTIONS",
            derived_from="la_ga_ratio",
            value_source="parsed_from_extracted",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="ga_fraction",
            value=ga_fraction,
            rule_id="R_LAGA_PARSE_FRACTIONS",
            derived_from="la_ga_ratio",
            value_source="parsed_from_extracted",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="LA/GA",
            value=la_over_ga,
            rule_id="R_LAGA_PARSE_OVER_GA",
            derived_from="la_ga_ratio",
            value_source="parsed_from_extracted",
            trace_pointer=trace_ptr,
        )

        mw_low, mw_high = parse_mw_range(get_field_value(row, "polymer_mw_kDa"))
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="polymer_mw_lower_kDa",
            value=mw_low,
            rule_id="R_POLYMER_MW_RANGE_PARSE",
            derived_from="polymer_mw_kDa",
            value_source="parsed_from_extracted",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="polymer_mw_upper_kDa",
            value=mw_high,
            rule_id="R_POLYMER_MW_RANGE_PARSE",
            derived_from="polymer_mw_kDa",
            value_source="parsed_from_extracted",
            trace_pointer=trace_ptr,
        )

        polymer_mass = parse_mass_to_mg(row.get("plga_mass_mg", ""))
        drug_mass = parse_mass_to_mg(row.get("drug_feed_amount_text", ""))
        drug_polymer_ratio = None
        if polymer_mass is not None and polymer_mass != 0 and drug_mass is not None:
            drug_polymer_ratio = drug_mass / polymer_mass

        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="polymer_mass_mg",
            value=polymer_mass,
            rule_id="R_POLYMER_MASS_PARSE",
            derived_from="plga_mass_mg",
            value_source="parsed_from_extracted",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="drug_mass_mg",
            value=drug_mass,
            rule_id="R_DRUG_MASS_PARSE",
            derived_from="drug_feed_amount_text",
            value_source="parsed_from_extracted",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="drug/polymer",
            value=drug_polymer_ratio,
            rule_id="R_DRUG_POLYMER_RATIO_COMPLETE",
            derived_from="drug_mass_mg,polymer_mass_mg",
            value_source="derived_math",
            trace_pointer=trace_ptr,
        )

        # v1 policy: derive aqueous/organic only from explicit evidence span with extracted anchors.
        span = row.get("evidence_span_text", "")
        has_anchor = any(str(row.get(col, "")).strip() != "" for col in anchors_for_ao)
        if has_anchor and str(span).strip():
            w1_o, wtot_o = parse_aqueous_organic_from_span(span)
            add_value(
                derived_rows,
                run_id=run_id,
                group_key=group_key,
                key=key,
                formulation_id=formulation_id,
                field_name="w1_over_o_ratio",
                value=w1_o,
                rule_id="R_AQUEOUS_ORGANIC_PARSE_W1_OVER_O",
                derived_from="evidence_span_text",
                value_source="parsed_evidence_span",
                trace_pointer=trace_ptr,
            )
            add_value(
                derived_rows,
                run_id=run_id,
                group_key=group_key,
                key=key,
                formulation_id=formulation_id,
                field_name="w1w2_over_o_ratio",
                value=wtot_o,
                rule_id="R_AQUEOUS_ORGANIC_PARSE_W1W2_OVER_O",
                derived_from="evidence_span_text",
                value_source="parsed_evidence_span",
                trace_pointer=trace_ptr,
            )

    derived_df = pd.DataFrame(derived_rows)
    if derived_df.empty:
        derived_df = pd.DataFrame(
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

    doe_out_dir = out_dir / "derivation_v1"
    doe_result = derive_doe_coded_factors(
        run_id=run_id,
        extracted=extracted,
        derived=derived_df,
        sample_manifest_path=Path("data/cleaned/samples/sample_goren18.tsv"),
        key2txt_path=Path("data/cleaned/content_goren_2025/key2txt.tsv"),
        out_dir=doe_out_dir,
    )
    derived_df = doe_result["derived_df"]

    out_tsv = out_dir / "derived_values.tsv"
    derived_df.to_csv(out_tsv, sep="\t", index=False)

    summary = {
        "run_id": run_id,
        "input_tsv": str(input_tsv),
        "rule_registry": str(rule_registry),
        "derived_values_rows": int(len(derived_df)),
        "distinct_group_keys": int(derived_df["group_key"].nunique()) if len(derived_df) > 0 else 0,
        "output": str(out_tsv),
        "doe_decode": doe_result["summary"],
    }
    (out_dir / "derivation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
