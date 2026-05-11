#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import (
    DATA_CLEANED_INDEX_DIR,
    DATA_RESULTS_DIR,
    DEV15_LAYER2_IDENTITY_TSV,
    DEV15_LAYER2_SOURCE_WORKBOOK_XLSX,
    DEV15_LAYER3_SOURCE_WORKBOOK_XLSX,
    DEV15_LAYER3_VALUES_TSV,
)
from src.utils.run_id import resolve_results_write_target

L2_PAPER = "paper_level_layer2_compare_v1.tsv"
L2_FORM = "formulation_level_layer2_compare_v1.tsv"
L2_SUMMARY = "layer2_compare_summary_v1.md"
L3_FIELD = "field_level_layer3_compare_v1.tsv"
L3_PAPER = "paper_field_coverage_summary_v1.tsv"
L3_SUMMARY = "layer3_compare_summary_v1.md"
MODELING = "dev15_deterministic_two_step_modeling_readiness_v3.md"
DEFAULT_PRIOR_BASELINE_RUN = DATA_RESULTS_DIR / "20260415_23c14f0" / "08_dev15_deterministic_two_step_diag_v2"
DEFAULT_PRIOR_ROW_BINDING_RUN = DATA_RESULTS_DIR / "20260415_23c14f0" / "15_dev15_deterministic_two_step_diag_with_binding_v1"

CLOSE_DIFF = 0.20
CLOSE_MATCH = 0.80
MOD_DIFF = 0.50
MOD_MATCH = 0.50
NUM_TOL = 0.01

FIELD_MAP = {
    "polymer_mw_kDa": "polymer_mw_kDa",
    "la_ga_ratio": "la_ga_ratio_normalized",
    "surfactant_name": "surfactant_name",
    "surfactant_concentration": "surfactant_concentration_value",
    "organic_solvent": "solvent_name",
    "drug_name": "drug_name",
    "drug_feed_amount": "drug_mass_mg",
    "polymer_amount": "polymer_mass_mg",
    "drug_polymer_ratio": "drug_to_polymer_ratio_raw",
    "phase_ratio": "phase_ratio_raw",
    "encapsulation_efficiency_percent": "ee_percent",
    "loading_capacity_percent": "lc_percent",
    "particle_size_nm": "particle_size_nm",
    "pdi": "pdi",
    "zeta_potential_mV": "zeta_mV",
}
TEXT_FIELDS = {"surfactant_name", "organic_solvent", "drug_name"}
SUPPORTED_STATUSES = {"explicit_supported", "relation_carried_explicit"}
PRIORITY_MODELING_FIELDS = [
    "encapsulation_efficiency_percent",
    "particle_size_nm",
    "pdi",
    "zeta_potential_mV",
    "loading_capacity_percent",
    "polymer_mw_kDa",
    "la_ga_ratio",
    "polymer_amount",
    "drug_feed_amount",
    "drug_polymer_ratio",
    "surfactant_concentration",
    "phase_ratio",
]
CORE_SET_A = ["drug_name", "polymer_mw_kDa", "la_ga_ratio", "encapsulation_efficiency_percent"]
CORE_SET_B = CORE_SET_A + ["polymer_amount", "drug_feed_amount"]
CORE_SET_C = CORE_SET_A + ["surfactant_concentration", "drug_polymer_ratio"]


def nt(v: Any) -> str:
    return str(v or "").strip()


def tok(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", nt(v).lower())


def fnum(v: Any) -> float | None:
    s = nt(v).replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else None


def rr(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h, delimiter="\t"))


def wt(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow({k: str(row.get(k, "")) for k in fields})


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")


def run(command: list[str]) -> tuple[int, str, str]:
    c = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    return c.returncode, c.stdout.strip(), c.stderr.strip()


def parse_json_out(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text[text.rfind("{") :])


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def gt_val(row: dict[str, str], field: str) -> str:
    val = nt(row.get(FIELD_MAP[field]))
    if field == "surfactant_concentration":
        unit = nt(row.get("surfactant_concentration_unit"))
        if val and unit:
            return f"{val} {unit}"
    return val


def cmp_val(field: str, gt: str, pred: str) -> str:
    if not gt and not pred:
        return "both_blank"
    if not gt:
        return "gt_blank"
    if not pred:
        return "pred_blank"
    if field in TEXT_FIELDS or field in {"la_ga_ratio", "drug_polymer_ratio", "phase_ratio"}:
        return "match" if tok(gt) == tok(pred) else "mismatch"
    g, p = fnum(gt), fnum(pred)
    if g is None or p is None:
        return "unknown"
    return "match" if abs(g - p) <= max(abs(g) * NUM_TOL, 1e-9) else "mismatch"


def l2_class(gt_count: int, pred_count: int, matched: int) -> tuple[str, str]:
    diff = abs(pred_count - gt_count) / gt_count if gt_count else 0.0
    mr = matched / gt_count if gt_count else 0.0
    if diff <= CLOSE_DIFF and mr >= CLOSE_MATCH:
        return "close_to_gt", "count and identity gap stay within close-range thresholds"
    if diff <= MOD_DIFF and mr >= MOD_MATCH:
        return "moderate_gap", "paper shows partial overlap but still material identity/count loss"
    return "severe_gap", "paper remains far from GT on identity overlap or count drift"


def build_l2(final_rows, gt_rows, freeze_by_paper, baseline_by_paper, out_dir: Path) -> dict[str, Any]:
    pred_by_paper, gt_by_paper = defaultdict(list), defaultdict(list)
    pred_by_rep, gt_by_rep = {}, {}
    for row in final_rows:
        pk, rep = nt(row.get("key")), nt(row.get("representative_source_formulation_id"))
        pred_by_paper[pk].append(row)
        if rep:
            pred_by_rep[(pk, rep)] = row
    for row in gt_rows:
        pk, rep = nt(row.get("paper_key")), nt(row.get("seed_pred_representative_source_formulation_id"))
        gt_by_paper[pk].append(row)
        if rep:
            gt_by_rep[(pk, rep)] = row

    frows, prows, total_gt, total_pred = [], [], 0, 0
    for pk in sorted(set(pred_by_paper) | set(gt_by_paper)):
        gt_rep = {nt(r.get("seed_pred_representative_source_formulation_id")) for r in gt_by_paper.get(pk, [])} - {""}
        pr_rep = {nt(r.get("representative_source_formulation_id")) for r in pred_by_paper.get(pk, [])} - {""}
        matched = sorted(gt_rep & pr_rep)
        for rep in matched:
            frows.append({"paper_key": pk, "gt_formulation_id": nt(gt_by_rep[(pk, rep)].get("gt_formulation_id")), "seed_pred_representative_source_formulation_id": rep, "predicted_final_formulation_id": nt(pred_by_rep[(pk, rep)].get("final_formulation_id")), "match_status": "matched_by_representative_source_formulation_id", "strict_freeze_status": nt(freeze_by_paper.get(pk, {}).get("status")), "baseline_ready_status": nt(baseline_by_paper.get(pk, {}).get("classification"))})
        for rep in sorted(gt_rep - pr_rep):
            frows.append({"paper_key": pk, "gt_formulation_id": nt(gt_by_rep[(pk, rep)].get("gt_formulation_id")), "seed_pred_representative_source_formulation_id": rep, "predicted_final_formulation_id": "", "match_status": "missing_in_prediction", "strict_freeze_status": nt(freeze_by_paper.get(pk, {}).get("status")), "baseline_ready_status": nt(baseline_by_paper.get(pk, {}).get("classification"))})
        for rep in sorted(pr_rep - gt_rep):
            frows.append({"paper_key": pk, "gt_formulation_id": "", "seed_pred_representative_source_formulation_id": rep, "predicted_final_formulation_id": nt(pred_by_rep[(pk, rep)].get("final_formulation_id")), "match_status": "extra_in_prediction", "strict_freeze_status": nt(freeze_by_paper.get(pk, {}).get("status")), "baseline_ready_status": nt(baseline_by_paper.get(pk, {}).get("classification"))})
        gtc, prc, mc = len(gt_by_paper.get(pk, [])), len(pred_by_paper.get(pk, [])), len(matched)
        gap, rationale = l2_class(gtc, prc, mc)
        prows.append({"paper_key": pk, "gt_count": gtc, "predicted_count": prc, "extra_count": prc - mc, "missing_count": gtc - mc, "matched_count": mc, "strict_freeze_status": nt(freeze_by_paper.get(pk, {}).get("status")), "baseline_ready_status": nt(baseline_by_paper.get(pk, {}).get("classification")), "gap_class": gap, "short_rationale": rationale})
        total_gt += gtc
        total_pred += prc

    wt(out_dir / L2_FORM, ["paper_key", "gt_formulation_id", "seed_pred_representative_source_formulation_id", "predicted_final_formulation_id", "match_status", "strict_freeze_status", "baseline_ready_status"], frows)
    wt(out_dir / L2_PAPER, ["paper_key", "gt_count", "predicted_count", "extra_count", "missing_count", "matched_count", "strict_freeze_status", "baseline_ready_status", "gap_class", "short_rationale"], prows)
    close = sum(1 for r in prows if r["gap_class"] == "close_to_gt")
    moderate = sum(1 for r in prows if r["gap_class"] == "moderate_gap")
    severe = sum(1 for r in prows if r["gap_class"] == "severe_gap")
    (out_dir / L2_SUMMARY).write_text(
        "\n".join([
            "# Layer2 Compare Summary v1",
            "",
            "Diagnostic-only, not benchmark-valid final output.",
            "",
            "## Thresholds",
            f"- close_to_gt: diff_ratio <= {CLOSE_DIFF} and matched_ratio >= {CLOSE_MATCH}",
            f"- moderate_gap: diff_ratio <= {MOD_DIFF} and matched_ratio >= {MOD_MATCH}",
            "- severe_gap: otherwise",
            "",
            "## Totals",
            f"- total_gt_rows: `{total_gt}`",
            f"- total_predicted_rows: `{total_pred}`",
            f"- close_to_gt papers: `{close}`",
            f"- moderate_gap papers: `{moderate}`",
            f"- severe_gap papers: `{severe}`",
        ]) + "\n",
        encoding="utf-8",
    )
    return {"paper_rows": prows, "total_gt": total_gt, "total_pred": total_pred, "close": close, "moderate": moderate, "severe": severe}


def build_l3(step2_rows, evidence_rows, gt_rows, baseline_by_paper, out_dir: Path) -> dict[str, Any]:
    ev_by_ff = {(nt(r.get("final_formulation_id")), nt(r.get("field_name"))): r for r in evidence_rows}
    rep_by_final = {}
    for row in evidence_rows:
        ff, rep = nt(row.get("final_formulation_id")), nt(row.get("source_representative_formulation_id"))
        if ff and rep and ff not in rep_by_final:
            rep_by_final[ff] = rep
    step2_by_rep = {}
    for row in step2_rows:
        ff, rep, pk = nt(row.get("final_formulation_id")), rep_by_final.get(nt(row.get("final_formulation_id")), ""), nt(row.get("paper_key"))
        if rep:
            step2_by_rep[(pk, rep)] = row

    frows = []
    sums: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for gt in gt_rows:
        pk, rep, gfid = nt(gt.get("paper_key")), nt(gt.get("seed_pred_representative_source_formulation_id")), nt(gt.get("gt_formulation_id"))
        pred = step2_by_rep.get((pk, rep))
        pff = nt(pred.get("final_formulation_id")) if pred else ""
        for field in FIELD_MAP:
            gv = gt_val(gt, field)
            if not gv:
                continue
            pv = nt(pred.get(f"{field}_value")) if pred else ""
            ps = nt(pred.get(f"{field}_support_status")) if pred else "identity_missing"
            pe = nt(pred.get(f"{field}_evidence_source_type")) if pred else ""
            cs = cmp_val(field, gv, pv) if pred else "unknown_alignment"
            frows.append({"paper_key": pk, "gt_formulation_id": gfid, "seed_pred_representative_source_formulation_id": rep, "matched_final_formulation_id": pff, "field_name": field, "gt_value": gv, "predicted_value": pv, "predicted_support_status": ps, "predicted_evidence_source_type": pe, "comparison_status": cs, "baseline_ready_status": nt(baseline_by_paper.get(pk, {}).get("classification"))})
            c = sums[(pk, field)]
            c["total_gt_cells"] += 1
            if pv:
                c["predicted_nonblank_cells"] += 1
            if cs == "match":
                c["exact_or_normalized_match_count"] += 1
            if ps in {"blank_not_reported", "unsupported_text", "parse_failed", "identity_missing", ""} and not pv:
                c["unsupported_or_blank_count"] += 1
            if ps == "unresolved_table":
                c["unresolved_table_count"] += 1
            if ps == "relation_carried_explicit":
                c["relation_carried_explicit_count"] += 1

    prows = []
    agg: dict[str, Counter[str]] = defaultdict(Counter)
    for (pk, field), c in sorted(sums.items()):
        total = c["total_gt_cells"]
        prows.append({"paper_key": pk, "field_name": field, "total_gt_cells": total, "predicted_nonblank_cells": c["predicted_nonblank_cells"], "exact_or_normalized_match_count": c["exact_or_normalized_match_count"], "unsupported_or_blank_count": c["unsupported_or_blank_count"], "unresolved_table_count": c["unresolved_table_count"], "relation_carried_explicit_count": c["relation_carried_explicit_count"], "coverage_ratio": f"{(c['predicted_nonblank_cells']/total) if total else 0.0:.4f}"})
        for k, v in c.items():
            agg[field][k] += v

    wt(out_dir / L3_FIELD, ["paper_key", "gt_formulation_id", "seed_pred_representative_source_formulation_id", "matched_final_formulation_id", "field_name", "gt_value", "predicted_value", "predicted_support_status", "predicted_evidence_source_type", "comparison_status", "baseline_ready_status"], frows)
    wt(out_dir / L3_PAPER, ["paper_key", "field_name", "total_gt_cells", "predicted_nonblank_cells", "exact_or_normalized_match_count", "unsupported_or_blank_count", "unresolved_table_count", "relation_carried_explicit_count", "coverage_ratio"], prows)
    lines = [
        "# Layer3 Compare Summary v1",
        "",
        "Diagnostic-only, not benchmark-valid final output.",
        "",
        "## Numeric Match Rule",
        f"- numeric fields count as match when absolute difference <= max(|gt| * {NUM_TOL}, 1e-9)",
        "- text fields count as match only after lowercase + punctuation-stripping normalization",
        "",
        "## Priority Fields",
    ]
    for field in FIELD_MAP:
        c = agg[field]
        total = c["total_gt_cells"]
        cov = (c["predicted_nonblank_cells"] / total) if total else 0.0
        lines.append(f"- `{field}`: gt_cells=`{total}` predicted_nonblank=`{c['predicted_nonblank_cells']}` matches=`{c['exact_or_normalized_match_count']}` unresolved_table=`{c['unresolved_table_count']}` relation_carried=`{c['relation_carried_explicit_count']}` coverage_ratio=`{cov:.4f}`")
    (out_dir / L3_SUMMARY).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"agg": agg}


def supported_row(row: dict[str, str], field: str) -> bool:
    return nt(row.get(f"{field}_support_status")) in SUPPORTED_STATUSES


def supported_overlap_count(rows: list[dict[str, str]], fields: list[str]) -> int:
    return sum(1 for row in rows if all(supported_row(row, field) for field in fields))


def explicit_or_relation_counts(evidence_rows: list[dict[str, str]]) -> Counter[str]:
    return Counter(
        nt(r.get("field_name"))
        for r in evidence_rows
        if nt(r.get("support_status")) in SUPPORTED_STATUSES
    )


def nonblank_counts(step2_rows: list[dict[str, str]]) -> dict[str, int]:
    return {field: sum(1 for r in step2_rows if nt(r.get(f"{field}_value"))) for field in FIELD_MAP}


def paper_improvement_counts(
    before_rows: list[dict[str, str]],
    after_rows: list[dict[str, str]],
    fields: list[str],
) -> Counter[str]:
    before_by_id = {nt(r.get("final_formulation_id")): r for r in before_rows}
    improved: Counter[str] = Counter()
    for row in after_rows:
        ff = nt(row.get("final_formulation_id"))
        prev = before_by_id.get(ff, {})
        delta = 0
        for field in fields:
            if not nt(prev.get(f"{field}_value")) and nt(row.get(f"{field}_value")):
                delta += 1
        if delta:
            improved[nt(row.get("paper_key"))] += delta
    return improved


def build_modeling(
    step2_rows,
    l2_paper_rows,
    evidence_rows,
    out_dir: Path,
    *,
    baseline_run_dir: Path | None = None,
    row_binding_run_dir: Path | None = None,
) -> dict[str, Any]:
    total = len(step2_rows)
    ee = [r for r in step2_rows if nt(r.get("encapsulation_efficiency_percent_value"))]
    core = [r for r in step2_rows if all(supported_row(r, field) for field in CORE_SET_A)]
    counts = nonblank_counts(step2_rows)
    bottlenecks = sorted(counts.items(), key=lambda x: x[1])[:5]
    explicit_counts = explicit_or_relation_counts(evidence_rows)
    l2c = Counter(r["gap_class"] for r in l2_paper_rows)
    core_papers = Counter(nt(r.get("paper_key")) for r in core)
    if not ee or not core:
        judgment = "not yet usable"
    elif len(core) >= 50 and len(core_papers) >= 5 and l2c.get("severe_gap", 0) <= 3:
        judgment = "usable for broader modeling prototype"
    else:
        judgment = "usable for small modeling pilot"
    effect_lines: list[str] = []
    baseline_summary: dict[str, Any] = {}
    row_binding_summary: dict[str, Any] = {}
    base_rows: list[dict[str, str]] = []
    row_binding_rows: list[dict[str, str]] = []
    if baseline_run_dir is not None:
        baseline_step2 = baseline_run_dir / "step2_outputs" / "step2_value_backfill_table_v1.tsv"
        baseline_evi = baseline_run_dir / "step2_outputs" / "step2_value_backfill_evidence_v1.tsv"
        if baseline_step2.exists() and baseline_evi.exists():
            base_rows = rr(baseline_step2)
            base_evidence = rr(baseline_evi)
            baseline_summary = {
                "ee": sum(1 for r in base_rows if nt(r.get("encapsulation_efficiency_percent_value"))),
                "core": supported_overlap_count(base_rows, CORE_SET_A),
                "core_a": supported_overlap_count(base_rows, CORE_SET_A),
                "core_b": supported_overlap_count(base_rows, CORE_SET_B),
                "core_c": supported_overlap_count(base_rows, CORE_SET_C),
                "support_counts": dict(explicit_or_relation_counts(base_evidence)),
            }
    if row_binding_run_dir is not None:
        row_step2 = row_binding_run_dir / "step2_outputs" / "step2_value_backfill_table_v1.tsv"
        row_evi = row_binding_run_dir / "step2_outputs" / "step2_value_backfill_evidence_v1.tsv"
        if row_step2.exists() and row_evi.exists():
            row_binding_rows = rr(row_step2)
            row_binding_evidence = rr(row_evi)
            row_binding_summary = {
                "ee": sum(1 for r in row_binding_rows if nt(r.get("encapsulation_efficiency_percent_value"))),
                "core": supported_overlap_count(row_binding_rows, CORE_SET_A),
                "core_a": supported_overlap_count(row_binding_rows, CORE_SET_A),
                "core_b": supported_overlap_count(row_binding_rows, CORE_SET_B),
                "core_c": supported_overlap_count(row_binding_rows, CORE_SET_C),
                "support_counts": dict(explicit_or_relation_counts(row_binding_evidence)),
            }

    combined_core_a = supported_overlap_count(step2_rows, CORE_SET_A)
    combined_core_b = supported_overlap_count(step2_rows, CORE_SET_B)
    combined_core_c = supported_overlap_count(step2_rows, CORE_SET_C)

    row_binding_improved = paper_improvement_counts(base_rows, row_binding_rows, PRIORITY_MODELING_FIELDS) if base_rows and row_binding_rows else Counter()
    parameter_improved = paper_improvement_counts(row_binding_rows, step2_rows, PRIORITY_MODELING_FIELDS) if row_binding_rows else Counter()
    combined_improved = paper_improvement_counts(base_rows, step2_rows, PRIORITY_MODELING_FIELDS) if base_rows else Counter()

    effect_lines = [
        "## Effect of Combined Binding Units",
        f"- comparison_baseline_run: `{rel(baseline_run_dir)}`" if baseline_run_dir is not None else "- comparison_baseline_run: `not provided`",
        f"- comparison_row_binding_run: `{rel(row_binding_run_dir)}`" if row_binding_run_dir is not None else "- comparison_row_binding_run: `not provided`",
        f"- DEV15 total formulation rows: `{len(base_rows) if base_rows else total} -> {len(row_binding_rows) if row_binding_rows else total} -> {total}`",
        f"- EE nonblank rows: `{baseline_summary.get('ee', 0)} -> {row_binding_summary.get('ee', 0)} -> {len(ee)}`",
        f"- Minimum modeling-core rows: `{baseline_summary.get('core', 0)} -> {row_binding_summary.get('core', 0)} -> {len(core)}`",
        f"- Core Set A overlap: `{baseline_summary.get('core_a', 0)} -> {row_binding_summary.get('core_a', 0)} -> {combined_core_a}`",
        f"- Core Set B overlap: `{baseline_summary.get('core_b', 0)} -> {row_binding_summary.get('core_b', 0)} -> {combined_core_b}`",
        f"- Core Set C overlap: `{baseline_summary.get('core_c', 0)} -> {row_binding_summary.get('core_c', 0)} -> {combined_core_c}`",
        "",
        "### Support Counts",
    ]
    base_support = baseline_summary.get("support_counts", {})
    row_support = row_binding_summary.get("support_counts", {})
    for field in PRIORITY_MODELING_FIELDS:
        effect_lines.append(
            f"- `{field}`: `{base_support.get(field, 0)} -> {row_support.get(field, 0)} -> {explicit_counts.get(field, 0)}`"
        )
    effect_lines.extend(
        [
            "",
            "### Top Papers Improved",
            "- row binding:",
            *( [f"  - `{paper}`: `{count}` upgraded prioritized field cells" for paper, count in row_binding_improved.most_common(5)] or ["  - none"] ),
            "- parameter binding:",
            *( [f"  - `{paper}`: `{count}` upgraded prioritized field cells" for paper, count in parameter_improved.most_common(5)] or ["  - none"] ),
            "- combined binding:",
            *( [f"  - `{paper}`: `{count}` upgraded prioritized field cells" for paper, count in combined_improved.most_common(5)] or ["  - none"] ),
            "",
            "### Remaining Bottlenecks After Combined Binding",
            *[f"- `{f}` nonblank rows: `{c}`" for f, c in bottlenecks],
            "",
        ]
    )
    lines = [
        "# DEV15 Deterministic Two-Step Modeling Readiness v3",
        "",
        "Diagnostic-only, not benchmark-valid final output.",
        "",
        "## Counts",
        f"- total_step2_formulation_rows: `{total}`",
        f"- formulations_with_nonblank_EE: `{len(ee)}`",
        f"- formulations_with_minimum_modeling_core_fields: `{len(core)}`",
        f"- papers_with_minimum_modeling_core_fields: `{len(core_papers)}`",
        "",
        *effect_lines,
        "## Bottlenecks",
        *[f"- `{f}` nonblank rows: `{c}`" for f, c in bottlenecks],
        "",
        "## Main Limitation Readout",
        f"- Step 1 identity gaps: `{'material' if l2c.get('severe_gap', 0) else 'limited'}`",
        f"- Step 2 evidence binding gaps: `{'material' if not ee else 'partial'}`",
        f"- table-row binding gaps: `{'material' if explicit_counts.get('encapsulation_efficiency_percent', 0) == 0 else 'reduced_but_partial'}`",
        "- GT alignment gaps: material whenever Step 1 representative-id alignment is missing",
        "",
        "## Judgment",
        f"- overall_judgment: `{judgment}`",
    ]
    (out_dir / MODELING).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "total": total,
        "ee": len(ee),
        "core": len(core),
        "core_a": combined_core_a,
        "core_b": combined_core_b,
        "core_c": combined_core_c,
        "judgment": judgment,
        "explicit_counts": dict(explicit_counts),
        "baseline_summary": baseline_summary,
        "row_binding_summary": row_binding_summary,
    }


def run_context(
    run_dir: Path,
    manifest: Path,
    step1: dict[str, Any],
    base: dict[str, Any],
    row_binding: dict[str, Any],
    parameter_binding: dict[str, Any],
    step2: dict[str, Any],
    logs: list[dict[str, str]],
    prior_baseline_run_dir: Path,
    prior_row_binding_run_dir: Path,
) -> str:
    lines = [
        "# RUN_CONTEXT",
        "",
        "## 1. Run type",
        "",
        "- `intermediate_diagnostic_run`",
        "- `deterministic_two_step_dev15_diagnostic_run`",
        "",
        "## 2. Purpose",
        "",
        "- Run the deterministic two-step baseline on DEV15 and compare the outputs against authoritative Layer2 and Layer3 GT as a diagnostic audit.",
        "- This run is diagnostic-only and not benchmark-valid final output.",
        "",
        "## 3. Authority resolution",
        "",
        f"- scope_manifest: `{rel(manifest)}`",
        f"- authoritative_layer2_gt_tsv: `{rel(DEV15_LAYER2_IDENTITY_TSV)}`",
        f"- authoritative_layer3_gt_tsv: `{rel(DEV15_LAYER3_VALUES_TSV)}`",
        f"- pinned_layer2_workbook_lineage_source: `{rel(DEV15_LAYER2_SOURCE_WORKBOOK_XLSX)}`",
        f"- pinned_layer3_workbook_lineage_source: `{rel(DEV15_LAYER3_SOURCE_WORKBOOK_XLSX)}`",
        f"- prior_baseline_run_dir: `{rel(prior_baseline_run_dir)}`",
        f"- prior_row_binding_run_dir: `{rel(prior_row_binding_run_dir)}`",
        "",
        "## 4. Step runs",
        "",
        f"- step1_run_dir: `{rel(run_dir)}`",
        f"- step1_identity_freeze: `{step1.get('identity_freeze', '')}`",
        f"- step1_run_status: `{step1.get('run_status', '')}`",
        f"- baseline_ready_assessment_step2_eligible_total: `{base.get('step2_eligible_total', '')}`",
        f"- table_row_binding_resolved_row_local_total: `{row_binding.get('resolved_row_local_total', '')}`",
        f"- formulation_parameter_binding_resolved_total: `{parameter_binding.get('resolved_supported_rows', '')}`",
        f"- step2_run_dir: `{rel(Path(step2.get('step2_run_path', run_dir)))}`",
        "",
        "## 5. Commands",
        "",
    ]
    for item in logs:
        lines.extend(["```powershell", item["command"], "```", f"- exit_status: `{item['status']}`", ""])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DEV15 deterministic two-step baseline diagnostic comparison.")
    parser.add_argument("--manifest-tsv", type=Path, default=DATA_CLEANED_INDEX_DIR / "manifest_current.tsv")
    parser.add_argument("--execution-cue", default="deterministic_two_step_dev15_diagnostic")
    parser.add_argument("--prior-baseline-run-dir", type=Path, default=DEFAULT_PRIOR_BASELINE_RUN)
    parser.add_argument("--prior-row-binding-run-dir", type=Path, default=DEFAULT_PRIOR_ROW_BINDING_RUN)
    args = parser.parse_args()

    tgt = resolve_results_write_target(results_root=DATA_RESULTS_DIR, default_child_cue=args.execution_cue)
    run_dir = Path(tgt["run_dir"])
    Path(tgt["bucket_dir"]).mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, str]] = []

    cmd1 = [sys.executable, str(REPO_ROOT / "src" / "analysis" / "run_deterministic_step1_baseline_v1.py"), "--manifest-tsv", str(args.manifest_tsv.resolve()), "--run-dir", str(run_dir)]
    c1, o1, e1 = run(cmd1); logs.append({"command": " ".join(cmd1), "status": str(c1), "stdout": o1, "stderr": e1}); step1 = parse_json_out(o1)
    cmd2 = [sys.executable, str(REPO_ROOT / "src" / "analysis" / "assess_baseline_ready_identity_v1.py"), "--run-dir", str(run_dir)]
    c2, o2, e2 = run(cmd2); logs.append({"command": " ".join(cmd2), "status": str(c2), "stdout": o2, "stderr": e2}); base = parse_json_out(o2)
    cmd3 = [sys.executable, str(REPO_ROOT / "src" / "analysis" / "build_table_row_binding_unit_v1.py"), "--final-table-tsv", str(run_dir / "final_formulation_table_v1.tsv"), "--decision-trace-tsv", str(run_dir / "final_output_decision_trace_v1.tsv"), "--relation-records-tsv", str(run_dir / "formulation_relation_v1" / "formulation_relation_records_v1.tsv"), "--resolved-relation-fields-tsv", str(run_dir / "formulation_relation_v1" / "resolved_relation_fields_v1.tsv"), "--out-dir", str(run_dir)]
    eligible_manifest = run_dir / "eligible_scope_manifest.tsv"
    if eligible_manifest.exists():
        cmd3.extend(["--scope-manifest-tsv", str(eligible_manifest)])
    c3, o3, e3 = run(cmd3); logs.append({"command": " ".join(cmd3), "status": str(c3), "stdout": o3, "stderr": e3}); row_binding = parse_json_out(o3)
    cmd4 = [sys.executable, str(REPO_ROOT / "src" / "analysis" / "build_formulation_parameter_binding_unit_v1.py"), "--final-table-tsv", str(run_dir / "final_formulation_table_v1.tsv"), "--decision-trace-tsv", str(run_dir / "final_output_decision_trace_v1.tsv"), "--relation-records-tsv", str(run_dir / "formulation_relation_v1" / "formulation_relation_records_v1.tsv"), "--resolved-relation-fields-tsv", str(run_dir / "formulation_relation_v1" / "resolved_relation_fields_v1.tsv"), "--out-dir", str(run_dir)]
    if eligible_manifest.exists():
        cmd4.extend(["--scope-manifest-tsv", str(eligible_manifest)])
    c4, o4, e4 = run(cmd4); logs.append({"command": " ".join(cmd4), "status": str(c4), "stdout": o4, "stderr": e4}); parameter_binding = parse_json_out(o4)
    cmd5 = [
        sys.executable,
        str(REPO_ROOT / "src" / "analysis" / "run_deterministic_step2_baseline_v1.py"),
        "--step1-run-dir",
        str(run_dir),
        "--table-row-binding-tsv",
        str(run_dir / "table_row_binding_resolved_v1.tsv"),
        "--parameter-binding-tsv",
        str(run_dir / "formulation_parameter_binding_resolved_v1.tsv"),
    ]
    c5, o5, e5 = run(cmd5); logs.append({"command": " ".join(cmd5), "status": str(c5), "stdout": o5, "stderr": e5}); step2 = parse_json_out(o5)
    step2_dir = Path(step2["step2_run_path"])
    for name in ["step2_value_backfill_table_v1.tsv", "step2_value_backfill_evidence_v1.tsv", "step2_value_backfill_summary_v1.md", "RUN_CONTEXT.md", "command_execution_log_v1.json"]:
        copy_if_exists(step2_dir / name, run_dir / "step2_outputs" / name)

    final_rows = rr(run_dir / "final_formulation_table_v1.tsv")
    freeze_rows = rr(run_dir / "audit" / "identity_freeze_guardrail_v1" / "identity_freeze_summary_v1.tsv")
    baseline_rows = rr(run_dir / "analysis" / "baseline_ready_identity_assessment_v1.tsv")
    l2_gt = rr(DEV15_LAYER2_IDENTITY_TSV)
    l3_gt = rr(DEV15_LAYER3_VALUES_TSV)
    step2_rows = rr(step2_dir / "step2_value_backfill_table_v1.tsv")
    step2_evi = rr(step2_dir / "step2_value_backfill_evidence_v1.tsv")
    freeze_by = {nt(r.get("paper_key")): r for r in freeze_rows}
    base_by = {nt(r.get("paper_key")): r for r in baseline_rows}
    adir = run_dir / "analysis"; adir.mkdir(parents=True, exist_ok=True)

    l2 = build_l2(final_rows, l2_gt, freeze_by, base_by, adir)
    build_l3(step2_rows, step2_evi, l3_gt, base_by, adir)
    mdl = build_modeling(
        step2_rows,
        l2["paper_rows"],
        step2_evi,
        adir,
        baseline_run_dir=args.prior_baseline_run_dir.resolve(),
        row_binding_run_dir=args.prior_row_binding_run_dir.resolve(),
    )

    (run_dir / "RUN_CONTEXT.md").write_text(
        run_context(
            run_dir,
            args.manifest_tsv.resolve(),
            step1,
            base,
            row_binding,
            parameter_binding,
            step2,
            logs,
            args.prior_baseline_run_dir.resolve(),
            args.prior_row_binding_run_dir.resolve(),
        ),
        encoding="utf-8",
    )
    (run_dir / "command_execution_log_v1.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")

    prev_step2 = args.prior_baseline_run_dir.resolve() / "step2_outputs" / "step2_value_backfill_table_v1.tsv"
    prev_evi = args.prior_baseline_run_dir.resolve() / "step2_outputs" / "step2_value_backfill_evidence_v1.tsv"
    prev_rows = rr(prev_step2) if prev_step2.exists() else []
    prev_ee = sum(1 for r in prev_rows if nt(r.get("encapsulation_efficiency_percent_value")))
    prev_core = supported_overlap_count(prev_rows, CORE_SET_A)
    row_step2 = args.prior_row_binding_run_dir.resolve() / "step2_outputs" / "step2_value_backfill_table_v1.tsv"
    row_rows = rr(row_step2) if row_step2.exists() else []
    row_ee = sum(1 for r in row_rows if nt(r.get("encapsulation_efficiency_percent_value")))
    row_core = supported_overlap_count(row_rows, CORE_SET_A)
    print(json.dumps({
        "run_dir": str(run_dir),
        "dev15_papers_processed": len({nt(r.get('key')) for r in final_rows}),
        "total_predicted_formulation_rows": len(final_rows),
        "layer2_total_gt_rows": l2["total_gt"],
        "layer2_total_predicted_rows": l2["total_pred"],
        "ee_nonblank_baseline_row_combined": [prev_ee, row_ee, mdl["ee"]],
        "minimum_modeling_core_rows_baseline_row_combined": [prev_core, row_core, mdl["core"]],
        "core_set_a_overlap_baseline_row_combined": [supported_overlap_count(prev_rows, CORE_SET_A), supported_overlap_count(row_rows, CORE_SET_A), mdl["core_a"]],
        "core_set_b_overlap_baseline_row_combined": [supported_overlap_count(prev_rows, CORE_SET_B), supported_overlap_count(row_rows, CORE_SET_B), mdl["core_b"]],
        "core_set_c_overlap_baseline_row_combined": [supported_overlap_count(prev_rows, CORE_SET_C), supported_overlap_count(row_rows, CORE_SET_C), mdl["core_c"]],
        "overall_judgment": mdl["judgment"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
