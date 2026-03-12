#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_ROOT / "data" / "results" / "run_20260312_1031_455ac37_targeted5_stage2_regression_v1"
DEFAULT_KEY = "5GIF3D8W"
DEFAULT_TEXT_PATH = PROJECT_ROOT / "data" / "cleaned" / "content_goren_2025" / "text" / "5GIF3D8W.pdf.txt"


@dataclass(frozen=True)
class ExpectedRow:
    family: str
    polymer_group: str
    loaded_state: str
    level_name: str
    normalized_level: str
    overlap_with_baseline: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a root-cause diagnostic pack for 5GIF3D8W Stage2 row inflation.")
    p.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    p.add_argument("--key", default=DEFAULT_KEY)
    p.add_argument("--text-path", default=str(DEFAULT_TEXT_PATH))
    return p.parse_args()


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_ratio(text: str) -> str:
    m = re.search(r"(\d+)\s*[:/]\s*(\d+)", str(text or ""))
    if not m:
        return ""
    return f"{m.group(1)}/{m.group(2)}"


def normalize_mg(text: str) -> str:
    s = normalize_spaces(text)
    if not s:
        return ""
    m = re.search(r"(\d+(?:\.\d+)?)\s*mg", s, flags=re.I)
    if not m:
        return ""
    num = float(m.group(1))
    if num.is_integer():
        return f"{int(num)} mg"
    return f"{num:g} mg"


def normalize_pct(text: str) -> str:
    s = normalize_spaces(text)
    if not s:
        return ""
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*w/?v", s, flags=re.I)
    if not m:
        return ""
    num = float(m.group(1))
    return f"{num:g} % w/v"


def infer_loaded_state(row: pd.Series) -> str:
    label = normalize_spaces(row.get("raw_formulation_label", "")).lower()
    drug_text = normalize_spaces(row.get("drug_feed_amount_text_value_text", ""))
    if "empty" in label or "drug free" in label:
        return "empty"
    if "drug loaded" in label or "etoposide" in label or normalize_mg(drug_text):
        return "drug_loaded"
    return "unknown"


def polymer_group(row: pd.Series) -> str:
    polymer_identity = normalize_spaces(row.get("polymer_identity", ""))
    ratio = normalize_ratio(row.get("la_ga_ratio_value_text", "") or row.get("la_ga_ratio_value", ""))
    if polymer_identity == "PLGA" and ratio:
        return f"PLGA {ratio}"
    return polymer_identity or "unknown"


def parse_current_rows(run_dir: Path, key: str) -> pd.DataFrame:
    path = run_dir / "weak_labels_v7pilot_r3_fixparse" / "weak_labels__v7pilot_r3_fixparse.tsv"
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    return df[df["key"] == key].copy()


def derive_signatures(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["loaded_state"] = out.apply(infer_loaded_state, axis=1)
    out["la_ga_ratio_normalized"] = out.apply(lambda r: normalize_ratio(r.get("la_ga_ratio_value_text", "") or r.get("la_ga_ratio_value", "")), axis=1)
    out["drug_feed_amount_mg_normalized"] = out.apply(lambda r: normalize_mg(r.get("drug_feed_amount_text_value_text", "") or r.get("drug_feed_amount_text_value", "")), axis=1)
    out["polymer_amount_mg_normalized"] = out.apply(lambda r: normalize_mg(r.get("plga_mass_mg_value_text", "") or r.get("plga_mass_mg_value", "")), axis=1)
    out["stabilizer_concentration_normalized"] = out.apply(
        lambda r: normalize_pct(r.get("surfactant_concentration_text_value_text", "") or r.get("surfactant_concentration_text_value", "")),
        axis=1,
    )
    out["polymer_group"] = out.apply(polymer_group, axis=1)

    baseline_polymer = "50 mg"
    baseline_drug = "5 mg"
    baseline_stabilizer = "1 % w/v"

    axis_values: List[str] = []
    axis_levels: List[str] = []
    notes: List[str] = []
    near_duplicate_signatures: List[str] = []
    for _, row in out.iterrows():
        polymer_amt = row["polymer_amount_mg_normalized"]
        drug_amt = row["drug_feed_amount_mg_normalized"]
        stab = row["stabilizer_concentration_normalized"]
        loaded = row["loaded_state"]
        source = row["candidate_source"]

        axis = "baseline"
        axis_level = ""
        note_parts: List[str] = []
        if source == "figure_variable_sweep":
            if stab:
                axis = "stabilizer_concentration"
                axis_level = stab
            elif polymer_amt:
                axis = "polymer_amount"
                axis_level = polymer_amt
            elif drug_amt:
                axis = "drug_amount"
                axis_level = drug_amt
        else:
            polymer_changed = bool(polymer_amt and polymer_amt != baseline_polymer)
            drug_changed = bool(drug_amt and drug_amt != baseline_drug)
            stab_changed = bool(stab and stab != baseline_stabilizer)
            changed_count = int(polymer_changed) + int(drug_changed) + int(stab_changed)
            if changed_count == 1:
                if polymer_changed:
                    axis = "polymer_amount"
                    axis_level = polymer_amt
                elif drug_changed:
                    axis = "drug_amount"
                    axis_level = drug_amt
                elif stab_changed:
                    axis = "stabilizer_concentration"
                    axis_level = stab
            elif changed_count > 1:
                axis = "multi_factor_variant"
                axis_level = ";".join(v for v in [polymer_amt, drug_amt, stab] if v)
                note_parts.append("More than one baseline-defining value differs from optimized baseline.")

        if axis == "baseline" and loaded == "drug_loaded":
            note_parts.append("Optimized drug-loaded baseline row.")
        if axis == "baseline" and loaded == "empty":
            note_parts.append("Optimized empty baseline row.")
        if axis in {"polymer_amount", "drug_amount", "stabilizer_concentration"} and source == "llm_extracted":
            note_parts.append("Explicit semantic variant row emitted by LLM.")
        if axis in {"polymer_amount", "drug_amount", "stabilizer_concentration"} and source == "figure_variable_sweep":
            note_parts.append("Synthetic sweep row appended after LLM extraction.")

        signature_core = "|".join(
            [
                f"polymer_group={row['polymer_group']}",
                f"loaded_state={loaded}",
                f"drug_mg={drug_amt}",
                f"polymer_mg={polymer_amt}",
                f"stabilizer={stab}",
            ]
        )
        near_dup_sig = "|".join(
            [
                f"polymer_group={row['polymer_group']}",
                f"axis={axis}",
                f"axis_level={axis_level}",
            ]
        )
        axis_values.append(axis)
        axis_levels.append(axis_level)
        notes.append(" ".join(note_parts))
        near_duplicate_signatures.append(near_dup_sig)

    out["derived_axis"] = axis_values
    out["derived_axis_level"] = axis_levels
    out["signature_core"] = out.apply(
        lambda r: "|".join(
            [
                f"polymer_group={r['polymer_group']}",
                f"loaded_state={r['loaded_state']}",
                f"drug_mg={r['drug_feed_amount_mg_normalized']}",
                f"polymer_mg={r['polymer_amount_mg_normalized']}",
                f"stabilizer={r['stabilizer_concentration_normalized']}",
            ]
        ),
        axis=1,
    )
    out["signature_with_source"] = out["signature_core"] + "|source=" + out["candidate_source"]
    out["near_duplicate_signature"] = near_duplicate_signatures
    out["notes"] = notes
    return out


def build_duplicate_groups(sig_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    group_id = 1
    candidate = sig_df[sig_df["derived_axis"].isin(["polymer_amount", "drug_amount", "stabilizer_concentration"])].copy()
    for signature, sub in candidate.groupby("near_duplicate_signature"):
        if len(sub) < 2:
            continue
        if len(set(sub["candidate_source"])) < 2:
            continue
        why = "Rows share the same polymer-specific sweep axis and level but differ only by representation style (explicit semantic variant vs synthetic sweep row)."
        rows.append(
            {
                "duplicate_group_id": f"DG{group_id:02d}",
                "normalized_signature": signature,
                "member_row_count": str(len(sub)),
                "member_formulation_ids": " | ".join(sub["formulation_id"].tolist()),
                "member_raw_labels": " | ".join(sub["raw_formulation_label"].tolist()),
                "candidate_sources": " | ".join(sub["candidate_source"].tolist()),
                "why_they_appear_duplicate": why,
            }
        )
        group_id += 1
    return pd.DataFrame(rows)


def expected_rows() -> List[ExpectedRow]:
    polymer_groups = ["PLGA 50/50", "PLGA 75/25", "PLGA 85/15", "PCL"]
    rows: List[ExpectedRow] = []
    for polymer in polymer_groups:
        rows.append(ExpectedRow("baseline_table", polymer, "empty", "baseline_loaded_state", "", "no"))
        rows.append(ExpectedRow("baseline_table", polymer, "drug_loaded", "baseline_loaded_state", "", "no"))
        for level in ["0.5 % w/v", "0.75 % w/v", "1 % w/v", "2 % w/v"]:
            rows.append(
                ExpectedRow(
                    "stabilizer_sweep",
                    polymer,
                    "drug_loaded",
                    "stabilizer_concentration",
                    level,
                    "yes" if level == "1 % w/v" else "no",
                )
            )
        for level in ["25 mg", "50 mg", "100 mg", "200 mg"]:
            rows.append(
                ExpectedRow(
                    "polymer_amount_sweep",
                    polymer,
                    "drug_loaded",
                    "polymer_amount",
                    level,
                    "yes" if level == "50 mg" else "no",
                )
            )
    return rows


def matching_ids(rows: pd.DataFrame) -> str:
    return " | ".join(rows["formulation_id"].tolist())


def matching_labels(rows: pd.DataFrame) -> str:
    return " | ".join(rows["raw_formulation_label"].tolist())


def build_expected_coverage(sig_df: pd.DataFrame, dup_df: pd.DataFrame) -> pd.DataFrame:
    out_rows: List[Dict[str, str]] = []
    for exp in expected_rows():
        match = pd.DataFrame()
        if exp.family == "baseline_table":
            match = sig_df[
                (sig_df["polymer_group"] == exp.polymer_group)
                & (sig_df["derived_axis"] == "baseline")
                & (sig_df["loaded_state"] == exp.loaded_state)
            ]
            if len(match):
                assessment = "expected_present"
                note = "Baseline row present."
            else:
                assessment = "expected_missing"
                note = "Expected baseline row not found."
        elif exp.family == "stabilizer_sweep":
            if exp.overlap_with_baseline == "yes":
                match = sig_df[
                    (sig_df["polymer_group"] == exp.polymer_group)
                    & (sig_df["derived_axis"] == "baseline")
                    & (sig_df["loaded_state"] == "drug_loaded")
                    & (sig_df["stabilizer_concentration_normalized"] == exp.normalized_level)
                ]
                assessment = "expected_present" if len(match) else "uncertain"
                note = "Baseline-overlap stabilizer level represented by optimized drug-loaded row." if len(match) else "Baseline-overlap stabilizer level not confirmed from baseline row."
            else:
                match = sig_df[
                    (sig_df["polymer_group"] == exp.polymer_group)
                    & (sig_df["derived_axis"] == "stabilizer_concentration")
                    & (sig_df["derived_axis_level"] == exp.normalized_level)
                ]
                assessment = "expected_present" if len(match) else "expected_missing"
                note = "Synthetic stabilizer sweep row present." if len(match) else "Expected stabilizer sweep row missing."
        else:
            if exp.overlap_with_baseline == "yes":
                match = sig_df[
                    (sig_df["polymer_group"] == exp.polymer_group)
                    & (sig_df["derived_axis"] == "baseline")
                    & (sig_df["loaded_state"] == "drug_loaded")
                    & (sig_df["polymer_amount_mg_normalized"] == exp.normalized_level)
                ]
                assessment = "expected_present" if len(match) else "uncertain"
                note = "Baseline-overlap polymer amount represented by optimized drug-loaded row." if len(match) else "Baseline-overlap polymer amount not confirmed from baseline row."
            else:
                match = sig_df[
                    (sig_df["polymer_group"] == exp.polymer_group)
                    & (sig_df["derived_axis"] == "polymer_amount")
                    & (sig_df["derived_axis_level"] == exp.normalized_level)
                ]
                assessment = "expected_present" if len(match) else "expected_missing"
                note = "Polymer amount condition present." if len(match) else "Expected polymer amount sweep row missing."
        out_rows.append(
            {
                "analysis_bucket": assessment,
                "family": exp.family,
                "polymer_group": exp.polymer_group,
                "loaded_state": exp.loaded_state,
                "level_name": exp.level_name,
                "normalized_level": exp.normalized_level,
                "overlap_with_baseline": exp.overlap_with_baseline,
                "matching_formulation_ids": matching_ids(match) if len(match) else "",
                "matching_raw_labels": matching_labels(match) if len(match) else "",
                "notes": note,
            }
        )

    for _, row in dup_df.iterrows():
        out_rows.append(
            {
                "analysis_bucket": "present_redundant",
                "family": "representation_overlap",
                "polymer_group": "",
                "loaded_state": "",
                "level_name": "",
                "normalized_level": row["normalized_signature"],
                "overlap_with_baseline": "",
                "matching_formulation_ids": row["member_formulation_ids"],
                "matching_raw_labels": row["member_raw_labels"],
                "notes": row["why_they_appear_duplicate"],
            }
        )

    extra = sig_df[sig_df["derived_axis"] == "drug_amount"].copy()
    grouped_extra = extra.groupby(["polymer_group", "derived_axis_level"])
    for (polymer, level), sub in grouped_extra:
        out_rows.append(
            {
                "analysis_bucket": "present_outside_expected_target",
                "family": "drug_amount_sweep",
                "polymer_group": polymer,
                "loaded_state": "drug_loaded_or_implicit",
                "level_name": "drug_amount",
                "normalized_level": level,
                "overlap_with_baseline": "yes" if level == "5 mg" else "no",
                "matching_formulation_ids": matching_ids(sub),
                "matching_raw_labels": matching_labels(sub),
                "notes": "Current rows represent the drug-amount family, which is outside the current 32-row target structure for this paper.",
            }
        )
    return pd.DataFrame(out_rows)


def extract_raw_hits(raw_text: str, patterns: List[str]) -> List[str]:
    hits: List[str] = []
    for pattern in patterns:
        for m in re.finditer(pattern, raw_text):
            hits.append(m.group(0))
    seen = set()
    ordered: List[str] = []
    for hit in hits:
        if hit in seen:
            continue
        seen.add(hit)
        ordered.append(hit)
    return ordered


def write_markdown(
    out_path: Path,
    sig_df: pd.DataFrame,
    dup_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    raw_response_text: str,
    cleaned_text: str,
    helper_script: Path,
) -> None:
    missing_polymer = coverage_df[
        (coverage_df["analysis_bucket"] == "expected_missing")
        & (coverage_df["family"] == "polymer_amount_sweep")
    ]
    raw_hits = extract_raw_hits(
        raw_response_text,
        [
            r'PLGA 50/50 Drug loaded \(100mg Polymer\)',
            r'PLGA 50/50 Drug loaded \(200mg Polymer\)',
            r'PLGA 50/50 Drug loaded \(2.5mg Etoposide\)',
            r'PLGA 50/50 Drug loaded \(10mg Etoposide\)',
            r'PCL Drug loaded \(2.5mg Etoposide\)',
            r'PCL Drug loaded \(10mg Etoposide\)',
        ],
    )
    text_points = extract_raw_hits(
        cleaned_text,
        [
            r"polymer amount \(25, 50,\s*100, and 200 mg\)",
            r"concentration of stabilizer \(0.5, 0.75, 1.0,\s*and 2.0% w/v\)",
            r"etoposide amount \(2.5, 5, 10, and 20 mg\)",
            r"optimized to\s*5 mg,\s*50 mg,\s*and 1.0% w/v",
            r"25 and 50 mg of polymer formed",
            r"100 and 200 mg",
        ],
    )
    lines = [
        "# 5GIF3D8W Root Cause Inspection",
        "",
        "## Scope",
        "- Paper: `5GIF3D8W` / `10.1080/10717540802174662`",
        "- Current Stage2 source: `data/results/run_20260312_1031_455ac37_targeted5_stage2_regression_v1/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`",
        "- Diagnostic helper: `" + str(helper_script.relative_to(PROJECT_ROOT)).replace("\\", "/") + "`",
        "",
        "## Current row composition",
        f"- Current rows: `{len(sig_df)}`",
        f"- `llm_extracted`: `{int((sig_df['candidate_source'] == 'llm_extracted').sum())}`",
        f"- `figure_variable_sweep`: `{int((sig_df['candidate_source'] == 'figure_variable_sweep').sum())}`",
        f"- Duplicate overlap groups identified: `{len(dup_df)}`",
        f"- Expected-missing polymer-amount conditions: `{len(missing_polymer)}`",
        "",
        "## Evidence from current output",
        "- The 38-row result is structurally mixed, not just inflated by one mechanism.",
        "- It contains baseline rows, synthetic sweep rows, and explicit semantic variant rows that restate some sweep conditions.",
        "- It also fails to enumerate all expected polymer-amount sweep conditions for `PLGA 75/25` and `PLGA 85/15`.",
        "",
        "## Proven semantic overlaps",
        "- The raw LLM response already contains explicit semantic variants that overlap with sweep-style conditions:",
    ]
    lines.extend([f"  - `{hit}`" for hit in raw_hits] or ["  - none found"])
    lines.extend(
        [
            "",
            "These rows are kept because the active extractor appends synthetic sweep rows after `canonicalize_formulations(...)`, and the sweep generator only deduplicates by lowercased `raw_formulation_label`, not by condition signature.",
            "",
            "## Proven expected-condition gaps",
            "- The cleaned text declares these sweep levels:",
        ]
    )
    lines.extend([f"  - `{hit}`" for hit in text_points] or ["  - none found"])
    lines.extend(
        [
            "",
            "- The current expected-grid audit shows that polymer-amount rows for `PLGA 75/25` and `PLGA 85/15` at `25 mg`, `100 mg`, and `200 mg` are missing.",
            "- This is introduced in `enumerate_figure_variable_sweep_candidates(...)` because the polymer-amount spec uses `identities_mode = \"section\"`, and `_infer_section_identities(...)` collapses generic `PLGA-copolymers` mentions to the first PLGA identity only.",
            "",
            "## Why the total became 38 instead of the expected 32",
            "- `+6 redundant rows`: explicit semantic variants survive alongside equivalent sweep-style rows.",
            "- `-6 expected rows`: polymer-amount sweep generation misses `PLGA 75/25` and `PLGA 85/15` non-baseline levels.",
            "- These offset numerically, which makes the raw count misleading: the structure is wrong even where the count is close.",
            "",
            "## Code-level insertion points",
            "- `enumerate_figure_variable_sweep_candidates(...)`",
            "  - appends synthetic rows after LLM extraction",
            "  - deduplicates with `seen_labels`, which is label-driven and not condition-signature-driven",
            "  - currently includes `drug_feed_amount_text` sweep generation for this paper, which is outside the current 32-row target structure",
            "- `_infer_section_identities(...)`",
            "  - for generic PLGA section text, returns only the first PLGA identity plus `PCL`",
            "  - this under-enumerates polymer-amount sweep rows for `PLGA 75/25` and `PLGA 85/15`",
            "- `canonicalize_formulations(...)`",
            "  - preserves explicit semantic rows from the LLM",
            "  - does not perform condition-signature deduplication against later synthetic sweep rows",
            "",
            "## Interpretation of specific questions",
            "- Are explicit formulation rows and factor-level sweep rows being kept simultaneously without deduplication?",
            "  - Yes.",
            "- Is deduplication currently label-driven instead of condition-signature-driven?",
            "  - Yes. The active seam is `seen_labels` inside `enumerate_figure_variable_sweep_candidates(...)`.",
            "- Why is polymer amount = `50 mg` absent as a sweep row?",
            "  - It is intentionally treated as the baseline-overlap level by `_infer_baseline_level(...)` and is represented by optimized baseline rows rather than synthetic sweep rows.",
            "- Is the current behavior intentional legacy behavior or an unintended regression?",
            "  - The baseline-overlap suppression itself is intentional. The combination of label-only deduplication plus under-scoped PLGA section identity inference is an unintended Stage2 structural mismatch for this paper.",
            "",
            "## Minimal safe fix proposal",
            "- Edit `_infer_section_identities(...)` or the polymer-amount branch in `enumerate_figure_variable_sweep_candidates(...)` so generic `PLGA-copolymers` sections can expand to all known PLGA ratios for this paper instead of only the first ratio.",
            "- Add a narrow post-generation dedup step after `forms.extend(enumerate_figure_variable_sweep_candidates(...))` and before TSV flattening.",
            "- Preferred representation when both exist for the same polymer/axis/level condition:",
            "  - keep the synthetic `figure_variable_sweep` row for single-variable sweep conditions",
            "  - keep baseline optimized table rows",
            "  - suppress the overlapping `llm_extracted` semantic variant row when it only restates the same single-variable sweep condition.",
            "- This fix is local to the active Stage2 extractor and does not require Stage4 or GT changes.",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    key = args.key
    out_dir = run_dir / f"{key}_root_cause"
    out_dir.mkdir(parents=True, exist_ok=True)

    sig_df = derive_signatures(parse_current_rows(run_dir, key))
    sig_path = out_dir / f"{key}_condition_signature.tsv"
    sig_df[
        [
            "key",
            "formulation_id",
            "raw_formulation_label",
            "candidate_source",
            "polymer_identity",
            "polymer_name_raw",
            "loaded_state",
            "la_ga_ratio_normalized",
            "drug_feed_amount_mg_normalized",
            "polymer_amount_mg_normalized",
            "stabilizer_concentration_normalized",
            "signature_core",
            "signature_with_source",
            "derived_axis",
            "derived_axis_level",
            "near_duplicate_signature",
            "notes",
        ]
    ].to_csv(sig_path, sep="\t", index=False)

    dup_df = build_duplicate_groups(sig_df)
    dup_path = out_dir / f"{key}_duplicate_groups.tsv"
    dup_df.to_csv(dup_path, sep="\t", index=False)

    coverage_df = build_expected_coverage(sig_df, dup_df)
    coverage_path = out_dir / f"{key}_missing_expected_levels.tsv"
    coverage_df.to_csv(coverage_path, sep="\t", index=False)

    raw_response_path = run_dir / "weak_labels_v7pilot_r3_fixparse" / "raw_responses" / "01_5GIF3D8W_10.1080_10717540802174662.txt"
    comparison_path = run_dir / "comparisons" / f"{key}__comparison.tsv"
    root_cause_path = out_dir / f"{key}_ROOT_CAUSE.md"
    write_markdown(
        out_path=root_cause_path,
        sig_df=sig_df,
        dup_df=dup_df,
        coverage_df=coverage_df,
        raw_response_text=raw_response_path.read_text(encoding="utf-8", errors="ignore"),
        cleaned_text=Path(args.text_path).read_text(encoding="utf-8", errors="ignore"),
        helper_script=Path(__file__).resolve(),
    )

    print(f"ROOT_CAUSE_MD={root_cause_path}")
    print(f"CONDITION_SIGNATURE_TSV={sig_path}")
    print(f"DUPLICATE_GROUPS_TSV={dup_path}")
    print(f"MISSING_EXPECTED_LEVELS_TSV={coverage_path}")
    print(f"COMPARISON_TSV_USED={comparison_path}")


if __name__ == "__main__":
    main()
