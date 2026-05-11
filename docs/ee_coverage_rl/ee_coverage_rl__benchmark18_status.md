# EE Coverage RL: Benchmark18 Stability Status (2026-02-24)

## Scope and Inputs Reviewed
- `PROJECT_ARCHITECTURE.md` was not found; architecture context was taken from `project/2_ARCHITECTURE.md`.
- Reviewed:
  - `README.md`
  - `project/FEATURE_EE_COVERAGE_RL_SCOPE.md`
  - `docs/tool_index.md`
- Benchmark/eval artifacts located under:
  - `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/`
  - `data/benchmark/goren_2025/overlap_goren18_v1/`

## Most Recent 18-Paper Benchmark Run
- Run ID: `run_20260219_1623_780eb83_goren18_weaklabels_v1`
- Reason selected: latest `data/results` run explicitly tagged `goren18` and contains full stage4/stage5 benchmark outputs.

## Requested Coverage Summary
- Formulation-level EE records: **158**
  - Source: `ee_modeling_projection_prep_summary.tsv` (`group_keys_with_exactly_1_EE_record=158`)
- With loading proxy: **89**
  - Source: `ee_modeling_projection_prep_summary.tsv` (`rows_with_EE_and_any_loading_proxy=89`)
- With polymer identity + loading proxy:
  - **69** in current run outputs (`projected_to_curated.tsv`, computed as EE + (`drug/polymer` or `LC`) + (`LA/GA` or `polymer_MW`))
  - **68** in branch scope doc (`project/FEATURE_EE_COVERAGE_RL_SCOPE.md`)
  - Interpretation: small drift (+1) relative to documented snapshot.

## Field-Level Evidence Status (Available)
No direct `supported/contradicted/insufficient/empty` table is emitted in this run.
Available proxy from evidence gate (`qc_field_evidence_gate_summary__realigned.tsv`):
- Supported (evidence pass): **493 / 912** (54.06%)
- Insufficient/mismatched (evidence fail): **419 / 912** (45.94%)
- Contradicted: **not separately reported**
- Empty: **not separately reported**

## What Is Stable Enough
- Structure and overlap coverage are stable:
  - 18/18 DOI overlap present (`row_membership_summary.json`)
  - schema_v3 relaxed alignment: recall **0.6535**, precision **0.6694**, F1 **0.6614**.
- De-dup at EE target level is stable:
  - exactly-1-EE groups: **158**
  - >1 EE per group: **0**.
- Evidence alignment improved materially after realignment:
  - realignment success: **636/912 (69.74%)**
  - major fail-rate drops vs pre-realign for EE, size, plga_mass, pva_conc, LC.

## Main Failure Modes and Frequencies
Benchmark alignment failures (schema_v3, relaxed):
- `field_mismatch_or_insufficient_overlap`: **44**
- `unmatched_projected_row`: **41**

Manual QC failure taxonomy (`qc_manual_failure_classification.tsv`, n=30):
- `true_missing`: **13** (43.33%)
- `wrong_unit`: **10** (33.33%)
- `span_misaligned`: **7** (23.33%)

## Decision: Move to Modeling?
**Yes, with a strict record filter.**

Rationale:
- Core extraction structure, DOI coverage, and dedup behavior are stable on the 18-paper benchmark.
- Remaining error mass is now mostly evidence quality/unit/span issues, which are better handled by filtering and targeted post-processing than by another broad prompt retune.
- Additional prompt tuning is unlikely to materially improve modeling readiness relative to gains from deterministic gating.

## Recommended Modeling-Ready Filter (Operational)
At formulation-group level, keep records where all conditions hold:
1. `EE` present and numeric (`encapsulation_efficiency_percent` / projected `EE`).
2. Evidence gate passes for `EE` after realignment.
3. At least one loading proxy present:
   - derived `drug/polymer` OR `loading_content_percent`/`LC`.
4. Polymer identity present:
   - `LA/GA` preferred; allow `polymer_MW` as secondary identity.
5. Exclude rows flagged with manual-equivalent failure causes:
   - `wrong_unit`, `span_misaligned`, or unresolved `true_missing` evidence.

This filter is consistent with moving forward to modeling while minimizing contamination from known extraction artifacts.

## Added Metrics Table
- `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/benchmark18_status_key_metrics.tsv`
