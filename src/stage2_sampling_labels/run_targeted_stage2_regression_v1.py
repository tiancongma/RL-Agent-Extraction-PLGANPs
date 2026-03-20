#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.run_id import build_run_id, get_git_short_hash

ACTIVE_EXTRACTOR = PROJECT_ROOT / "src" / "stage2_sampling_labels" / "auto_extract_weak_labels_v7pilot_r3_fixparse.py"
PILOT3_MANIFEST = PROJECT_ROOT / "data" / "cleaned" / "goren_2025" / "index" / "splits" / "dev_manifest_v7pilot3_2026-03-06.tsv"
REMAINING12_MANIFEST = PROJECT_ROOT / "data" / "cleaned" / "goren_2025" / "index" / "splits" / "dev_manifest_remaining12_2026-03-10.tsv"
PILOT3_REFERENCE = PROJECT_ROOT / "data" / "results" / "run_20260310_v7pilot3r3fixparse_synthmethod" / "weak_labels_v7pilot_r3_fixparse" / "weak_labels__v7pilot_r3_fixparse.tsv"
REMAINING12_REFERENCE = PROJECT_ROOT / "data" / "results" / "run_20260310_dev15_remaining12_synthmethod_merged" / "weak_labels_v7pilot_r3_fixparse" / "weak_labels__v7pilot_r3_fixparse.tsv"

REQUIRED_PLGA_COLUMNS = [
    "la_ga_ratio_value",
    "la_ga_ratio_value_text",
    "polymer_mw_kDa_value",
    "polymer_mw_kDa_value_text",
    "plga_mass_mg_value",
    "plga_mass_mg_value_text",
]


@dataclass(frozen=True)
class PaperSpec:
    zotero_key: str
    role_in_regression_set: str
    selection_reason: str
    manifest_source: str
    reference_tsv: Path
    reference_note: str


PAPER_SPECS: List[PaperSpec] = [
    PaperSpec(
        zotero_key="5GIF3D8W",
        role_in_regression_set="mixed_polymer_case",
        selection_reason="Mixed-polymer stress case with prior Stage2 under-enumeration, later figure-sweep recovery, and current explicit polymer identity expectations.",
        manifest_source="remaining12",
        reference_tsv=REMAINING12_REFERENCE,
        reference_note="Prior comparable DEV15 Stage2 remaining12 merged output.",
    ),
    PaperSpec(
        zotero_key="5ZXYABSU",
        role_in_regression_set="plga_only_stable_baseline",
        selection_reason="PLGA-only stable baseline used to check additive polymer fields do not perturb row counts, labels, or PLGA-family exports.",
        manifest_source="pilot3",
        reference_tsv=PILOT3_REFERENCE,
        reference_note="Prior comparable DEV15 pilot3 Stage2 synthmethod output.",
    ),
    PaperSpec(
        zotero_key="L3H2RS2H",
        role_in_regression_set="boundary_fragile_case",
        selection_reason="Known segmentation/alignment-fragile formulation boundary case with prior under-segmentation risk.",
        manifest_source="pilot3",
        reference_tsv=PILOT3_REFERENCE,
        reference_note="Prior comparable DEV15 pilot3 Stage2 synthmethod output.",
    ),
    PaperSpec(
        zotero_key="WFDTQ4VX",
        role_in_regression_set="doe_multifactor_case",
        selection_reason="DOE-style multi-factor case used to check that additive polymer identity does not disturb multi-row sweep grouping.",
        manifest_source="remaining12",
        reference_tsv=REMAINING12_REFERENCE,
        reference_note="Prior comparable DEV15 Stage2 remaining12 merged output.",
    ),
    PaperSpec(
        zotero_key="WIVUCMYG",
        role_in_regression_set="table_dominant_multisource_case",
        selection_reason="Table-dominant high-row paper with additional text/html evidence, useful for checking row preservation and audit-readiness.",
        manifest_source="pilot3",
        reference_tsv=PILOT3_REFERENCE,
        reference_note="Prior comparable DEV15 pilot3 Stage2 synthmethod output.",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a targeted 5-paper regression for the active Stage2 extractor.")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--max-chars", type=int, default=50000)
    p.add_argument("--sleep", type=float, default=0.4)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--version", type=int, default=1)
    return p.parse_args()


def read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def git_branch() -> str:
    out = subprocess.check_output(
        ["git", "branch", "--show-current"],
        cwd=PROJECT_ROOT,
        text=True,
    )
    return out.strip() or "detached"


def resolve_selected_manifest() -> pd.DataFrame:
    pilot_df = read_manifest(PILOT3_MANIFEST)
    remaining_df = read_manifest(REMAINING12_MANIFEST)
    pieces: List[pd.DataFrame] = []
    for spec in PAPER_SPECS:
        source_df = remaining_df if spec.manifest_source == "remaining12" else pilot_df
        hit = source_df[source_df["key"] == spec.zotero_key]
        if len(hit) != 1:
            raise RuntimeError(f"Expected exactly one manifest row for {spec.zotero_key} in {spec.manifest_source}.")
        pieces.append(hit.copy())
    selected = pd.concat(pieces, ignore_index=True)
    return selected[["key", "doi", "title", "text_path"]]


def write_tsv(path: Path, rows: List[Dict[str, str]], columns: List[str]) -> None:
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, sep="\t", index=False)


def build_run_context(
    run_id: str,
    branch: str,
    model: str,
    run_dir: Path,
    output_tsv: Path,
) -> str:
    bullet_keys = "\n".join(
        f"- `{spec.zotero_key}`: {spec.selection_reason}" for spec in PAPER_SPECS
    )
    return f"""# RUN_CONTEXT

## 1. Run ID
`{run_id}`

## 2. Run type
`component_regression_run`

Benchmark reporting rule:
- This run is `diagnostic-only, not benchmark-valid final output`.
- Only `full_pipeline_benchmark_run` may report official GT comparison results.

## 3. Date / branch / active script path
- Date: `{datetime.now().isoformat(timespec='seconds')}`
- Branch: `{branch}`
- Active script: `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
- Model: `{model}`

## 4. Why this run exists
- This targeted regression checks whether the current active Stage2 path remains stable after recent Stage2 fixes, repository reduction, active-path clarification, and the additive polymer identity schema extension.

## 5. Recent code and architecture changes relevant to this run
- Figure-derived formulation-variable sweeps were added back as low-confidence Stage2 candidates.
- Mixed-polymer extraction scope was de-biased so explicit non-PLGA rows are retained.
- Additive extraction-layer fields `polymer_identity` and `polymer_name_raw` were added without removing PLGA-specific fields.
- Active-path governance was reconciled to the reduced repository structure.

## 6. Why full DEV15 was NOT run first
- Full DEV15 rerun was deferred because the current risk is regression from recent targeted fixes, not lack of broader coverage.
- A 5-paper targeted run gives faster feedback on mixed-polymer behavior, boundary stability, DOE grouping, and table-heavy row preservation before spending time on a full rerun.

## 7. Paper selection strategy
- Use five papers chosen by failure mode and audit value rather than chronology.
- Include the required mixed-polymer and PLGA-only anchors.
- Add one boundary-fragile case, one DOE-style case, and one table-dominant multi-source case from existing DEV15 notes and manual review summaries.

## 8. Selected paper keys and rationale for each
{bullet_keys}

## 9. Explicit test objectives
- Check formulation row counts for collapse or explosion versus prior comparable Stage2 outputs.
- Verify `polymer_identity` and `polymer_name_raw` populate plausibly.
- Confirm mixed-polymer rows remain separated.
- Confirm existing PLGA-specific exports remain present.
- Check that synthetic sweep rows still export and remain auditable.

## 10. Required comparison checks
- Compare each paper against the most relevant prior DEV15 Stage2 output already checked into the repo.
- Review row counts, polymer distributions, unknown polymer counts, and label/grouping deltas.
- Flag any missing PLGA-specific columns as an immediate regression.

## 11. Expected pass criteria
- No obvious unintended row collapse or row explosion.
- `polymer_identity` populated plausibly with limited explainable `unknown`.
- Existing PLGA-specific fields still exported.
- No evidence that additive polymer identity changed formulation grouping unexpectedly.

## 12. Output files generated by this run
- `{run_dir / 'RUN_CONTEXT.md'}`
- `{run_dir / 'paper_selection.tsv'}`
- `{run_dir / 'targeted_manifest.tsv'}`
- `{output_tsv}`
- `{run_dir / 'regression_summary.tsv'}`
- `{run_dir / 'regression_notes.md'}`
- `{run_dir / 'comparisons'}` (per-paper comparison TSVs when prior comparable outputs exist)

## 13. Recommended next action depending on outcome
- If all five papers remain stable and mixed-polymer behavior looks correct: proceed to full DEV15 rerun.
- If any paper shows unexplained row collapse, row inflation, or polymer grouping drift: stop and fix Stage2 before full DEV15.
"""


def stringify_counter(counter: Counter) -> str:
    ordered = {k: counter[k] for k in sorted(counter)}
    return json.dumps(ordered, ensure_ascii=False, sort_keys=True)


def safe_jsonish(s: str) -> str:
    return str(s or "").strip()


def build_label_key(df: pd.DataFrame) -> pd.Series:
    if "raw_formulation_label" in df.columns:
        vals = df["raw_formulation_label"].fillna("").astype(str).str.strip()
        return vals.where(vals != "", df["formulation_id"].fillna("").astype(str).str.strip())
    return df["formulation_id"].fillna("").astype(str).str.strip()


def compare_labels(current_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    cur = current_df.copy()
    base = baseline_df.copy()
    cur["label_key"] = build_label_key(cur)
    base["label_key"] = build_label_key(base)
    keep_cols = ["label_key", "formulation_id", "raw_formulation_label", "candidate_source", "polymer_identity", "polymer_name_raw"]
    for col in keep_cols:
        if col not in cur.columns:
            cur[col] = ""
        if col not in base.columns:
            base[col] = ""
    cur = cur[keep_cols].drop_duplicates(subset=["label_key"])
    base = base[keep_cols].drop_duplicates(subset=["label_key"])
    merged = base.merge(cur, on="label_key", how="outer", suffixes=("_baseline", "_current"), indicator=True)
    merged["comparison_status"] = merged["_merge"].map(
        {
            "both": "shared_label",
            "left_only": "baseline_only",
            "right_only": "current_only",
        }
    )
    return merged.drop(columns=["_merge"]).sort_values(["comparison_status", "label_key"]).reset_index(drop=True)


def assess_row_count(spec: PaperSpec, n_current: int, n_baseline: int) -> str:
    if n_current == n_baseline:
        return f"Stable versus prior comparable Stage2 output ({n_current} rows)."
    if spec.zotero_key == "5GIF3D8W" and n_current == 32 and n_baseline == 6:
        return "Expected recovery from prior 6-row under-enumeration to 32 mixed-polymer + figure-sweep candidates."
    if spec.zotero_key == "WFDTQ4VX" and abs(n_current - n_baseline) <= 1:
        return f"Near-stable DOE count versus prior Stage2 output ({n_baseline} -> {n_current}); review grouping but not an immediate failure."
    return f"Count shifted versus prior comparable Stage2 output ({n_baseline} -> {n_current}); inspect comparison file."


def assess_grouping(spec: PaperSpec, current_df: pd.DataFrame, compare_df: pd.DataFrame) -> str:
    current_only = int((compare_df["comparison_status"] == "current_only").sum())
    baseline_only = int((compare_df["comparison_status"] == "baseline_only").sum())
    sweep_count = int((current_df.get("candidate_source", "") == "figure_variable_sweep").sum())
    if spec.zotero_key == "5GIF3D8W":
        return f"Mixed-polymer case now keeps explicit PCL rows and {sweep_count} synthetic figure-sweep rows; compare current_only={current_only}, baseline_only={baseline_only}."
    if current_only == 0 and baseline_only == 0:
        return "No label-level grouping drift against prior comparable output."
    return f"Label-level drift detected: current_only={current_only}, baseline_only={baseline_only}."


def assess_polymer(current_df: pd.DataFrame) -> str:
    polymer_col = current_df.get("polymer_identity")
    raw_col = current_df.get("polymer_name_raw")
    if polymer_col is None or raw_col is None:
        return "Missing additive polymer fields."
    unknown = int((polymer_col.fillna("").astype(str).str.strip().str.lower() == "unknown").sum())
    distinct = sorted({x for x in polymer_col.fillna("").astype(str).str.strip() if x})
    raw_populated = int((raw_col.fillna("").astype(str).str.strip() != "").sum())
    return f"polymer_identity families={distinct}; unknown={unknown}; polymer_name_raw populated on {raw_populated}/{len(current_df)} rows."


def assess_status(spec: PaperSpec, current_df: pd.DataFrame, n_current: int, n_baseline: int, missing_plga_cols: List[str]) -> str:
    if missing_plga_cols:
        return "FAIL"
    unknown = int((current_df["polymer_identity"].fillna("").astype(str).str.strip().str.lower() == "unknown").sum())
    if spec.zotero_key == "5GIF3D8W":
        if n_current == 32 and unknown == 0:
            return "PASS"
        return "FAIL"
    if n_current == n_baseline and unknown <= max(1, len(current_df) // 5):
        return "PASS"
    if spec.zotero_key == "WFDTQ4VX" and abs(n_current - n_baseline) <= 1 and unknown <= max(1, len(current_df) // 5):
        return "WARN"
    return "WARN"


def main() -> None:
    args = parse_args()
    branch = git_branch()
    git_hash = get_git_short_hash(PROJECT_ROOT)
    run_id = build_run_id(subset="targeted5", stage="stage2_regression", version=args.version, git_hash=git_hash)
    run_dir = PROJECT_ROOT / "data" / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir = run_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    selected_manifest = resolve_selected_manifest()
    manifest_path = run_dir / "targeted_manifest.tsv"
    selected_manifest.to_csv(manifest_path, sep="\t", index=False)

    paper_selection_rows = [
        {
            "zotero_key": spec.zotero_key,
            "role_in_regression_set": spec.role_in_regression_set,
            "selection_reason": spec.selection_reason,
        }
        for spec in PAPER_SPECS
    ]
    write_tsv(
        run_dir / "paper_selection.tsv",
        paper_selection_rows,
        ["zotero_key", "role_in_regression_set", "selection_reason"],
    )

    out_dir = run_dir / "weak_labels_v7pilot_r3_fixparse"
    output_tsv = out_dir / "weak_labels__v7pilot_r3_fixparse.tsv"
    run_context = build_run_context(run_id=run_id, branch=branch, model=args.model, run_dir=run_dir, output_tsv=output_tsv)
    (run_dir / "RUN_CONTEXT.md").write_text(run_context, encoding="utf-8")

    print(f"run_id={run_id}")
    print("selected_papers:")
    for spec in PAPER_SPECS:
        print(f"- {spec.zotero_key}: {spec.selection_reason}")

    cmd = [
        sys.executable,
        str(ACTIVE_EXTRACTOR),
        "--manifest-tsv",
        str(manifest_path),
        "--model",
        args.model,
        "--max-items",
        str(len(PAPER_SPECS)),
        "--max-chars",
        str(args.max_chars),
        "--sleep",
        str(args.sleep),
        "--retries",
        str(args.retries),
        "--out-dir",
        str(out_dir),
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing_pythonpath else f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}"
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)

    current_df = pd.read_csv(output_tsv, sep="\t", dtype=str).fillna("")
    missing_plga_cols = [c for c in REQUIRED_PLGA_COLUMNS if c not in current_df.columns]

    summary_rows: List[Dict[str, str]] = []
    notes_lines = [
        "# Regression Notes",
        "",
        f"- Run ID: `{run_id}`",
        f"- Active extractor: `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`",
        f"- Model: `{args.model}`",
        "",
        "## Findings by paper",
    ]

    overall_fail = False
    overall_warn = False

    for spec in PAPER_SPECS:
        cur = current_df[current_df["key"] == spec.zotero_key].copy()
        baseline_df = pd.read_csv(spec.reference_tsv, sep="\t", dtype=str).fillna("")
        base = baseline_df[baseline_df["key"] == spec.zotero_key].copy()
        compare_df = compare_labels(cur, base)
        compare_df.to_csv(comparisons_dir / f"{spec.zotero_key}__comparison.tsv", sep="\t", index=False)

        n_current = len(cur)
        n_baseline = len(base)
        polymer_counts = Counter(x for x in cur["polymer_identity"].astype(str).str.strip() if x)
        unknown_polymer_count = int(polymer_counts.get("unknown", 0))
        row_count_comment = assess_row_count(spec, n_current, n_baseline)
        grouping_comment = assess_grouping(spec, cur, compare_df)
        polymer_comment = assess_polymer(cur)
        status = assess_status(spec, cur, n_current, n_baseline, missing_plga_cols)

        if status == "FAIL":
            overall_fail = True
        elif status == "WARN":
            overall_warn = True

        summary_rows.append(
            {
                "zotero_key": spec.zotero_key,
                "n_rows_current": str(n_current),
                "polymer_identity_counts": stringify_counter(polymer_counts),
                "unknown_polymer_count": str(unknown_polymer_count),
                "row_count_comment": row_count_comment,
                "grouping_comment": grouping_comment,
                "polymer_comment": polymer_comment,
                "regression_status": status,
            }
        )

        notes_lines.extend(
            [
                f"### {spec.zotero_key}",
                f"- Role: `{spec.role_in_regression_set}`",
                f"- Prior comparable output: `{spec.reference_tsv.relative_to(PROJECT_ROOT)}`",
                f"- Current rows: `{n_current}`",
                f"- Prior rows: `{n_baseline}`",
                f"- Polymer counts: `{stringify_counter(polymer_counts)}`",
                f"- Status: `{status}`",
                f"- Row count comment: {row_count_comment}",
                f"- Grouping comment: {grouping_comment}",
                f"- Polymer comment: {polymer_comment}",
                "",
            ]
        )

    summary_path = run_dir / "regression_summary.tsv"
    write_tsv(
        summary_path,
        summary_rows,
        [
            "zotero_key",
            "n_rows_current",
            "polymer_identity_counts",
            "unknown_polymer_count",
            "row_count_comment",
            "grouping_comment",
            "polymer_comment",
            "regression_status",
        ],
    )

    if missing_plga_cols:
        notes_lines.extend(
            [
                "## Global field check",
                f"- Missing required PLGA-specific output columns: `{', '.join(missing_plga_cols)}`",
                "",
            ]
        )
    else:
        notes_lines.extend(
            [
                "## Global field check",
                "- Required polymer/shared output columns remain present: `la_ga_ratio_*`, `polymer_mw_kDa_*`, `plga_mass_mg_*`.",
                "",
            ]
        )

    recommendation = "NEED_FIX_BEFORE_DEV15" if overall_fail or overall_warn else "GO_TO_DEV15"
    notes_lines.extend(
        [
            "## Recommendation",
            f"- `{recommendation}`",
        ]
    )
    (run_dir / "regression_notes.md").write_text("\n".join(notes_lines), encoding="utf-8")

    print(f"RUN_CONTEXT={run_dir / 'RUN_CONTEXT.md'}")
    print(f"SUMMARY_TSV={summary_path}")
    print(f"RECOMMENDATION={recommendation}")


if __name__ == "__main__":
    main()
