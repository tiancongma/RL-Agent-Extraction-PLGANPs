#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_sampling_labels.auto_extract_weak_labels_v7pilot_r3_fixparse import (  # noqa: E402
    _extract_declared_levels,
    _extract_section_window,
    _infer_polymer_identities,
    _infer_section_identities,
)

DEFAULT_PRE_RUN = PROJECT_ROOT / "data" / "results" / "run_20260312_1031_455ac37_targeted5_stage2_regression_v1"
DEFAULT_POST_RUN = PROJECT_ROOT / "data" / "results" / "run_20260312_1253_455ac37_targeted5_stage2_regression_v1"
DEFAULT_TEXT_PATH = PROJECT_ROOT / "data" / "cleaned" / "content_goren_2025" / "text" / "5GIF3D8W.pdf.txt"
KEY = "5GIF3D8W"


@dataclass(frozen=True)
class AxisSpec:
    axis: str
    field_name: str
    field_label: str
    decl_pattern: str
    start_patterns: List[str]
    end_patterns: List[str]
    identities_mode: str
    preferred_supported_groups: List[str]


SPECS: List[AxisSpec] = [
    AxisSpec(
        axis="stabilizer_concentration",
        field_name="surfactant_concentration_text",
        field_label="stabilizer concentration",
        decl_pattern=r"concentration of stabilizer\s*\((.*?)\)",
        start_patterns=[r"The effect of stabilizer concentration", r"Effect of stabilizer concentration"],
        end_patterns=[r"\bPolymer Content\b", r"\bAmount of Drug\b", r"\bIn Vitro Drug Release\b"],
        identities_mode="all",
        preferred_supported_groups=["PLGA 50/50", "PLGA 75/25", "PLGA 85/15", "PCL"],
    ),
    AxisSpec(
        axis="polymer_amount",
        field_name="plga_mass_mg",
        field_label="polymer amount",
        decl_pattern=r"polymer amount\s*\((.*?)\)",
        start_patterns=[r"\bPolymer Content\b"],
        end_patterns=[r"\bAmount of Drug\b", r"\bIn Vitro Drug Release\b"],
        identities_mode="section",
        preferred_supported_groups=["PLGA 50/50", "PLGA 75/25", "PLGA 85/15", "PCL"],
    ),
    AxisSpec(
        axis="drug_amount",
        field_name="drug_feed_amount_text",
        field_label="etoposide amount",
        decl_pattern=r"etoposide amount\s*\((.*?)\)",
        start_patterns=[r"\bAmount of Drug\b"],
        end_patterns=[r"\bIn Vitro Drug Release\b", r"\bFIG\.\s*4\b", r"\bFIG\.\s*5\b"],
        identities_mode="section",
        preferred_supported_groups=["PLGA 50/50", "PCL"],
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose axis-specific applicability for 5GIF3D8W in the post-fix Stage2 run.")
    p.add_argument("--pre-run-dir", default=str(DEFAULT_PRE_RUN))
    p.add_argument("--post-run-dir", default=str(DEFAULT_POST_RUN))
    p.add_argument("--text-path", default=str(DEFAULT_TEXT_PATH))
    return p.parse_args()


def load_rows(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "weak_labels_v7pilot_r3_fixparse" / "weak_labels__v7pilot_r3_fixparse.tsv"
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    return df[df["key"] == KEY].copy()


def normalize_spaces(text: str) -> str:
    return " ".join(str(text or "").split())


def normalize_ratio(text: str) -> str:
    import re

    m = re.search(r"(\d+)\s*[:/]\s*(\d+)", str(text or ""))
    if not m:
        return ""
    return f"{m.group(1)}/{m.group(2)}"


def normalize_mg(text: str) -> str:
    import re

    s = normalize_spaces(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*mg", s, flags=re.I)
    if not m:
        return ""
    val = float(m.group(1))
    return f"{val:g} mg"


def normalize_pct(text: str) -> str:
    import re

    s = normalize_spaces(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*w/?v", s, flags=re.I)
    if not m:
        return ""
    val = float(m.group(1))
    return f"{val:g} % w/v"


def infer_loaded_state(row: pd.Series) -> str:
    label = normalize_spaces(row.get("raw_formulation_label", "")).lower()
    drug_text = normalize_spaces(row.get("drug_feed_amount_text_value_text", "") or row.get("drug_feed_amount_text_value", ""))
    if "empty" in label or "empty formulation" in drug_text.lower():
        return "empty"
    if "drug loaded" in label or normalize_mg(drug_text):
        return "drug_loaded"
    return "unknown"


def polymer_group(row: pd.Series) -> str:
    polymer_identity = normalize_spaces(row.get("polymer_identity", ""))
    ratio = normalize_ratio(row.get("la_ga_ratio_value_text", "") or row.get("la_ga_ratio_value", ""))
    if polymer_identity == "PLGA" and ratio:
        return f"PLGA {ratio}"
    return polymer_identity or "unknown"


def classify_axis(row: pd.Series) -> Tuple[str, str]:
    source = row.get("candidate_source", "")
    if source == "figure_variable_sweep":
        stab = normalize_pct(row.get("surfactant_concentration_text_value_text", "") or row.get("surfactant_concentration_text_value", ""))
        if stab:
            return "stabilizer_concentration", stab
        polym = normalize_mg(row.get("plga_mass_mg_value_text", "") or row.get("plga_mass_mg_value", ""))
        if polym and polym != "50 mg":
            return "polymer_amount", polym
        drug = normalize_mg(row.get("drug_feed_amount_text_value_text", "") or row.get("drug_feed_amount_text_value", ""))
        if drug and drug != "5 mg":
            return "drug_amount", drug
        return "other_or_uncertain", ""
    return "baseline_table", ""


def build_row_annotations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["loaded_state"] = out.apply(infer_loaded_state, axis=1)
    out["la_ga_ratio_normalized"] = out.apply(lambda r: normalize_ratio(r.get("la_ga_ratio_value_text", "") or r.get("la_ga_ratio_value", "")), axis=1)
    out["polymer_group"] = out.apply(polymer_group, axis=1)
    axis_info = out.apply(classify_axis, axis=1)
    out["sweep_axis"] = [a for a, _ in axis_info]
    out["sweep_level_normalized"] = [b for _, b in axis_info]
    out["inferred_section_identity_source"] = out["candidate_source"].map(
        lambda x: "synthetic_axis_section_inference" if x == "figure_variable_sweep" else "llm_baseline_table_row"
    )
    notes: List[str] = []
    for _, row in out.iterrows():
        note_parts: List[str] = []
        if row["sweep_axis"] == "baseline_table":
            note_parts.append("Baseline or optimized table-origin row.")
        elif row["sweep_axis"] == "drug_amount":
            note_parts.append("Drug-amount sweep row.")
        elif row["sweep_axis"] == "polymer_amount":
            note_parts.append("Polymer-amount sweep row.")
        elif row["sweep_axis"] == "stabilizer_concentration":
            note_parts.append("Stabilizer-concentration sweep row.")
        notes.append(" ".join(note_parts))
    out["notes"] = notes
    return out


def extract_axis_evidence(raw_text: str, forms: List[Dict[str, Any]], spec: AxisSpec) -> Dict[str, Any]:
    section_text, section_start, section_end = _extract_section_window(
        raw_text,
        start_patterns=spec.start_patterns,
        end_patterns=spec.end_patterns,
        pre_context=120,
    )
    declared_levels = _extract_declared_levels(raw_text, spec.decl_pattern)
    all_identities = _infer_polymer_identities(forms, raw_text)
    inferred_section_identities = (
        list(all_identities)
        if spec.identities_mode == "all"
        else _infer_section_identities(section_text, all_identities)
    )
    safe_snippet = section_text[:1400].encode("ascii", "replace").decode("ascii").replace("\n", " ")
    lower = section_text.lower()
    evidence_basis_parts: List[str] = []
    if "plga-copolymers" in lower or "plga copolymers" in lower:
        evidence_basis_parts.append("generic_PLGA_copolymers_wording")
    if "pcl" in lower:
        evidence_basis_parts.append("explicit_PCL_mention")
    if "plga 50/50" in lower:
        evidence_basis_parts.append("explicit_PLGA50_50_mention")
    if "plga 75/25" in lower:
        evidence_basis_parts.append("explicit_PLGA75_25_mention")
    if "plga 85/15" in lower:
        evidence_basis_parts.append("explicit_PLGA85_15_mention")
    if spec.axis == "drug_amount":
        if "for plga 50/50" in lower:
            evidence_basis_parts.append("numeric_results_for_PLGA50_50")
        if "for pcl" in lower:
            evidence_basis_parts.append("numeric_results_for_PCL")
    if spec.axis == "polymer_amount":
        if "forplga50/50" in lower or "for plga 50/50" in lower:
            evidence_basis_parts.append("explicit_polymer_amount_result_PLGA50_50")
        if "formulations prepared with pcl" in lower:
            evidence_basis_parts.append("explicit_polymer_amount_result_PCL")
    return {
        "axis": spec.axis,
        "section_start": section_start,
        "section_end": section_end,
        "declared_levels": declared_levels,
        "all_identities": all_identities,
        "inferred_section_identities": inferred_section_identities,
        "safe_snippet": safe_snippet,
        "evidence_basis": ", ".join(evidence_basis_parts),
    }


def build_axis_matrix(axis_evidence: Dict[str, Dict[str, Any]], post_rows: pd.DataFrame) -> pd.DataFrame:
    polymer_groups = ["PLGA 50/50", "PLGA 75/25", "PLGA 85/15", "PCL"]
    rows: List[Dict[str, str]] = []
    for spec in SPECS:
        generated = set(
            post_rows[post_rows["sweep_axis"] == spec.axis]["polymer_group"].tolist()
        )
        evidence = axis_evidence[spec.axis]
        snippet = evidence["safe_snippet"]
        for polymer_group in polymer_groups:
            evidence_status = "unsupported"
            evidence_type = "unsupported_expansion"
            diagnostic_comment = ""
            if spec.axis in {"stabilizer_concentration", "polymer_amount"}:
                if polymer_group.startswith("PLGA "):
                    evidence_status = "supported"
                    evidence_type = "shared_polymer_family_wording"
                    diagnostic_comment = "Section uses PLGA-copolymer family wording that can reasonably cover all PLGA ratios for this axis."
                if polymer_group == "PCL":
                    evidence_status = "supported"
                    evidence_type = "explicit_section_local_support"
                    diagnostic_comment = "Section mentions PCL explicitly for this axis."
            elif spec.axis == "drug_amount":
                if polymer_group == "PLGA 50/50":
                    evidence_status = "supported"
                    evidence_type = "explicit_section_local_support"
                    diagnostic_comment = "Drug-amount section reports numeric effect explicitly for PLGA 50/50."
                elif polymer_group == "PCL":
                    evidence_status = "supported"
                    evidence_type = "explicit_section_local_support"
                    diagnostic_comment = "Drug-amount section reports numeric effect explicitly for PCL."
                else:
                    evidence_status = "unsupported"
                    evidence_type = "inferred_only"
                    diagnostic_comment = "Drug-amount section does not explicitly name this PLGA ratio; current generation appears to come from reused shared PLGA identity expansion."
            current_generation_status = "generated" if polymer_group in generated else "not_generated"
            rows.append(
                {
                    "polymer_group": polymer_group,
                    "sweep_axis": spec.axis,
                    "evidence_support_status": evidence_status,
                    "evidence_type": evidence_type,
                    "evidence_basis": evidence["evidence_basis"],
                    "current_generation_status": current_generation_status,
                    "diagnostic_comment": diagnostic_comment,
                }
            )
    return pd.DataFrame(rows)


def build_axis_evidence_map(axis_evidence: Dict[str, Dict[str, Any]], post_rows: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for spec in SPECS:
        ev = axis_evidence[spec.axis]
        generated_groups = sorted(set(post_rows[post_rows["sweep_axis"] == spec.axis]["polymer_group"].tolist()))
        rows.append(
            {
                "sweep_axis": spec.axis,
                "declared_levels": " | ".join(ev["declared_levels"]),
                "section_window_span": f"{ev['section_start']}:{ev['section_end']}",
                "explicitly_named_polymer_groups_in_section": _explicit_named_groups(ev["safe_snippet"]),
                "generic_family_language_present": _generic_family_flags(ev["safe_snippet"]),
                "current_section_inference_result": " | ".join(ev["inferred_section_identities"]),
                "current_generated_polymer_groups": " | ".join(generated_groups),
                "evidence_basis": ev["evidence_basis"],
                "section_snippet_ascii": ev["safe_snippet"],
            }
        )
    return pd.DataFrame(rows)


def _explicit_named_groups(snippet: str) -> str:
    groups: List[str] = []
    lower = snippet.lower()
    for group in ["PLGA 50/50", "PLGA 75/25", "PLGA 85/15", "PCL"]:
        if group.lower() in lower:
            groups.append(group)
    return " | ".join(groups)


def _generic_family_flags(snippet: str) -> str:
    flags: List[str] = []
    lower = snippet.lower()
    if "plga-copolymers" in lower or "plga copolymers" in lower:
        flags.append("PLGA_copolymer_family")
    if "all the polymers in the study" in lower:
        flags.append("all_polymers_in_study")
    return " | ".join(flags)


def build_overexpanded_rows(post_rows: pd.DataFrame, matrix_df: pd.DataFrame, pre_rows: pd.DataFrame) -> pd.DataFrame:
    pre_labels = set(pre_rows["raw_formulation_label"].tolist())
    unsupported = matrix_df[matrix_df["evidence_support_status"] == "unsupported"][["polymer_group", "sweep_axis"]]
    unsupported_pairs = {(r["polymer_group"], r["sweep_axis"]) for _, r in unsupported.iterrows()}
    rows: List[Dict[str, str]] = []
    for _, row in post_rows.iterrows():
        pair = (row["polymer_group"], row["sweep_axis"])
        if pair not in unsupported_pairs:
            continue
        raw_label = row["raw_formulation_label"]
        newly_added = "yes" if raw_label not in pre_labels else "no"
        confidence = "high" if newly_added == "yes" else "medium"
        rows.append(
            {
                "formulation_id": row["formulation_id"],
                "raw_formulation_label": raw_label,
                "polymer_identity": row["polymer_group"],
                "sweep_axis": row["sweep_axis"],
                "sweep_level": row["sweep_level_normalized"],
                "why_likely_overexpanded": "Axis-local section evidence does not directly support this polymer group, but current generation inherited a broadened shared PLGA identity set.",
                "what_evidence_is_missing": "No explicit section-local naming or numeric evidence for this polymer group on this axis.",
                "confidence_of_diagnostic": confidence,
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    out_path: Path,
    pre_rows: pd.DataFrame,
    post_rows: pd.DataFrame,
    matrix_df: pd.DataFrame,
    over_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
) -> None:
    new_rows = sorted(set(post_rows["raw_formulation_label"]) - set(pre_rows["raw_formulation_label"]))
    lines = [
        "# 5GIF3D8W Axis Applicability Root Cause",
        "",
        "## 1. What changed between the first targeted run and the post-fix rerun",
        f"- Pre-fix run rows: `{len(pre_rows)}`",
        f"- Post-fix run rows: `{len(post_rows)}`",
        "- Newly added rows after the last fix:",
    ]
    lines.extend([f"  - `{label}`" for label in new_rows] or ["  - none"])
    lines.extend(
        [
            "",
            "## 2. Why the old duplicate seam is no longer the main problem",
            "- The post-fix run no longer shows the explicit semantic-vs-synthetic overlap groups that drove the earlier 38-row mixed structure.",
            "- The remaining issue is axis-specific applicability drift: `drug_amount` sweep rows are now generated for all four polymer groups even though the paper-local section evidence does not support all four equally.",
            "",
            "## 3. Axis-by-axis evidence interpretation",
        ]
    )
    for axis in ["stabilizer_concentration", "polymer_amount", "drug_amount"]:
        sub = matrix_df[matrix_df["sweep_axis"] == axis]
        lines.append(f"### {axis}")
        for _, row in sub.iterrows():
            lines.append(
                f"- `{row['polymer_group']}`: `{row['evidence_support_status']}` / `{row['evidence_type']}` / current=`{row['current_generation_status']}`. {row['diagnostic_comment']}"
            )
        lines.append("")
    lines.extend(
        [
            "## 4. Which rows are likely unsupported expansions",
            f"- Rows flagged as likely over-expanded: `{len(over_df)}`",
        ]
    )
    for _, row in over_df.iterrows():
        lines.append(
            f"- `{row['raw_formulation_label']}`: axis=`{row['sweep_axis']}`, polymer=`{row['polymer_identity']}`, confidence=`{row['confidence_of_diagnostic']}`."
        )
    lines.extend(
        [
            "",
            "## 5. Where axis specificity is lost in code",
            "- `_infer_section_identities(...)` is axis-agnostic: it receives only `section_text` plus a fallback identity pool and does not retain which sweep axis is being evaluated.",
            "- `enumerate_figure_variable_sweep_candidates(...)` calls `_infer_section_identities(...)` for both `polymer_amount` and `drug_amount` with the same section-level widening logic.",
            "- After the earlier fix, generic PLGA family wording became sufficient to widen to all known PLGA identities, which is reasonable for `polymer_amount` but too broad for `drug_amount` in this paper.",
            "- The `drug_amount` section window also carries nearby generic formulation language via the current extraction windowing strategy, so axis-local applicability is not preserved separately from shared identity inference.",
            "",
            "## 6. The narrowest recommended future fix",
            "- Constrain identity widening separately by sweep axis inside `enumerate_figure_variable_sweep_candidates(...)`.",
            "- Keep the current `polymer_amount` shared-PLGA expansion intact.",
            "- Do not change baseline row handling or the existing overlap-dedup seam.",
            "- For `drug_amount`, require explicit axis-local support before widening beyond the directly evidenced polymer groups.",
            "- Regression risk to watch after the future fix: avoid reintroducing the previous missing `polymer_amount` rows for `PLGA 75/25` and `PLGA 85/15` while tightening `drug_amount` applicability.",
            "",
            "## 7. Next-step status",
            "- `READY_FOR_AXIS_SCOPING_FIX`",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    pre_run = Path(args.pre_run_dir)
    post_run = Path(args.post_run_dir)
    out_dir = post_run / f"{KEY}_axis_applicability_root_cause"
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_rows_raw = load_rows(pre_run)
    post_rows_raw = load_rows(post_run)
    pre_rows = build_row_annotations(pre_rows_raw)
    post_rows = build_row_annotations(post_rows_raw)

    raw_text = Path(args.text_path).read_text(encoding="utf-8", errors="ignore")
    forms_for_identity = []
    for _, row in post_rows_raw.iterrows():
        forms_for_identity.append(
            {
                "raw_formulation_label": row["raw_formulation_label"],
                "polymer_identity": row["polymer_identity"],
                "fields": {
                    "la_ga_ratio": {
                        "value_text": row.get("la_ga_ratio_value_text", ""),
                        "value": row.get("la_ga_ratio_value", ""),
                    }
                },
            }
        )

    axis_evidence = {spec.axis: extract_axis_evidence(raw_text, forms_for_identity, spec) for spec in SPECS}
    matrix_df = build_axis_matrix(axis_evidence, post_rows)
    evidence_df = build_axis_evidence_map(axis_evidence, post_rows)
    over_df = build_overexpanded_rows(post_rows, matrix_df, pre_rows)

    row_annotation_path = out_dir / f"{KEY}_axis_row_annotations.tsv"
    post_rows[
        [
            "key",
            "formulation_id",
            "raw_formulation_label",
            "candidate_source",
            "polymer_identity",
            "la_ga_ratio_normalized",
            "loaded_state",
            "sweep_axis",
            "sweep_level_normalized",
            "inferred_section_identity_source",
            "notes",
        ]
    ].to_csv(row_annotation_path, sep="\t", index=False)

    matrix_path = out_dir / f"{KEY}_axis_applicability_matrix.tsv"
    matrix_df.to_csv(matrix_path, sep="\t", index=False)

    over_path = out_dir / f"{KEY}_axis_overexpanded_rows.tsv"
    over_df.to_csv(over_path, sep="\t", index=False)

    evidence_path = out_dir / f"{KEY}_axis_evidence_map.tsv"
    evidence_df.to_csv(evidence_path, sep="\t", index=False)

    md_path = out_dir / f"{KEY}_AXIS_APPLICABILITY_ROOT_CAUSE.md"
    write_markdown(md_path, pre_rows, post_rows, matrix_df, over_df, evidence_df)

    print(f"ROOT_CAUSE_MD={md_path}")
    print(f"AXIS_APPLICABILITY_MATRIX_TSV={matrix_path}")
    print(f"AXIS_OVEREXPANDED_ROWS_TSV={over_path}")
    print(f"AXIS_EVIDENCE_MAP_TSV={evidence_path}")
    print(f"ROW_ANNOTATIONS_TSV={row_annotation_path}")


if __name__ == "__main__":
    main()
