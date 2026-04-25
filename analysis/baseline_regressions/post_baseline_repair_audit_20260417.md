# Post-Baseline Repair Audit: 20260417 / 385b6e1

## Facts

### Audit anchor
- Baseline commit: `385b6e19b6dd9737ef2b1135c36fedc3be41c37d`
- Baseline run:
  `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1`
- Post-baseline commit range inspected: `385b6e1..HEAD`
- `git diff --name-status 385b6e1..HEAD -- project docs src analysis data/results/20260417_385b6e1`
  returned `22` tracked-file changes in the inspected surfaces.

### Change inventory

#### Maintained script changes
- `src/stage1_cleaning/pdf2clean.py`
  - covered by pattern: `PAT_STAGE2_SCOPE_BINDING_REFRESH_V1`
- `src/stage2_sampling_labels/run_stage2_composite_v1.py`
  - covered by pattern: `PAT_STAGE2_SCOPE_BINDING_REFRESH_V1`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  - covered by pattern: `PAT_WFDTQ4VX_SELECTOR_RESTORE_V1`
  - supporting negative guard: `PAT_SORTED_CSV_FIRST4_MULTI_TABLE_FALLBACK_V1`
- `src/stage2_sampling_labels/table_row_expansion_v1.py`
  - covered by pattern: `PAT_TABLE_SCOPE_ID_SYNTHESIS_V1`
- `src/stage5_benchmark/enforce_identity_freeze_v1.py`
  - covered by pattern: `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1`
- `src/stage5_benchmark/compare_final_table_to_gt_v1.py`
  - covered by pattern: `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1`

#### Governance-support changes
- `project/2_ARCHITECTURE.md`
  - covered but non-pattern governance support: `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1`
- `project/ACTIVE_PIPELINE_FLOW.md`
  - covered but non-pattern governance support: `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
  - covered but non-pattern governance support: `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1`
- `docs/feature_governance/daily_audit_scope_contract_v1.tsv`
  - covered but non-pattern governance support only

#### Added audit or analysis support surfaces
- Stage1 portability and Stage2 path-verification artifacts under `analysis/` and `docs/audits/`
  - covered by pattern: `PAT_STAGE2_SCOPE_BINDING_REFRESH_V1`
- `analysis/stage1_validation/dev15_clean_text_completeness.tsv`
  - covered by pattern: `PAT_STAGE2_SCOPE_BINDING_REFRESH_V1`
- `analysis/stage1_validation/dev15_local_source_mapping_validation.md`
  - covered by pattern: `PAT_STAGE2_SCOPE_BINDING_REFRESH_V1`
- `analysis/stage1_validation/dev15_local_zotero_dual_source_audit.tsv`
  - covered by pattern: `PAT_STAGE2_SCOPE_BINDING_REFRESH_V1`
- `docs/audits/dev15_baseline_governance_membership_audit_2026-04-17.md`
  - covered but non-pattern governance support: `governance_support_change`
  - governed reason: the file audits baseline classification and registration posture, but it does not by itself prove successful activation of a maintained repair unit

#### Materially relevant post-baseline runs
- `data/results/20260417_385b6e1/10_5gif3d8w_capability_restoration_validation_v1`
  - covered by pattern family: `PAT_5GIF3D8W_SWEEP_AUTH_PRESERVATION_V1`
- `data/results/20260417_385b6e1/10_qlyk_capability_restoration_v1`
  - covered but non-pattern: `supporting_run_without_pattern`
  - reason: artifacts exist, but no `RUN_CONTEXT.md`
- `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2`
  - covered by pattern: `PAT_5GIF3D8W_SWEEP_AUTH_PRESERVATION_V1`
- `data/results/20260417_385b6e1/11_qlyk_capability_restoration_bounded_v1`
  - covered by pattern support: `PAT_QLYKLPKT_TWO_TABLE_PRESERVATION_V1`
- `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1`
  - covered by pattern: `PAT_QLYKLPKT_TWO_TABLE_PRESERVATION_V1`

## Acceptance Findings

| finding_id | acceptance finding | result | classification | linked index status | evidence |
|---|---|---|---|---|---|
| `AF001` | `10_qlyk_capability_restoration_v1` was covered in the audit but not mapped to any pattern | resolved conservatively | `covered-but-non-pattern`; `supporting_run_without_pattern`; `missing-run-surface` | linked to `PAT_QLYKLPKT_TWO_TABLE_PRESERVATION_V1` as non-compliant supporting run only | `find .../10_qlyk_capability_restoration_v1 -name RUN_CONTEXT.md` returned none |
| `AF002` | Maintained-entrypoint changes to `enforce_identity_freeze_v1.py` and `compare_final_table_to_gt_v1.py` were absent from the repair index | resolved | `covered-by-pattern`; `governance_support_change` | new `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1` | `11_5gif.../gt_authority_v2_variantaware/RUN_CONTEXT.md`; Stage5 diff snippets |
| `AF003` | `src/stage2_sampling_labels/table_row_expansion_v1.py` was absent from the repair index | resolved with conservative status | `covered-by-pattern` | new `PAT_TABLE_SCOPE_ID_SYNTHESIS_V1` as `candidate_historical_pattern` | post-baseline diff plus `12_qlyk.../compatibility_projection_summary_v1.json` |
| `AF004` | `PAT_WFDTQ4VX_SELECTOR_RESTORE_V1` was labeled `active_mainline_pattern` without explicit activation evidence | resolved | `status-overclaim` | downgraded to `candidate_historical_pattern` | baseline `09/RUN_CONTEXT.md` only states purpose; no run-local feature proves WFDTQ4VX-specific activation |
| `AF005` | `PAT_UFXX9WXE_DOE_MAINLINE_RESTORE_V1` cited a validating run surface that does not exist | resolved | `missing-run-surface`; `status-overclaim` | downgraded to `candidate_historical_pattern` | `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix` is missing |
| `AF006` | Governance file changes were covered in audit but not linked in the repair index | resolved | `covered-but-non-pattern`; `governance_support_change` | linked to `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1` | `project/2_ARCHITECTURE.md`; `project/ACTIVE_PIPELINE_FLOW.md`; `project/ACTIVE_PIPELINE_RUNBOOK.md` |
| `AF007` | Four materially relevant tracked support files were still uncovered | resolved | `covered-by-pattern`; `covered-but-non-pattern` | `analysis/stage1_validation/dev15_clean_text_completeness.tsv`, `analysis/stage1_validation/dev15_local_source_mapping_validation.md`, and `analysis/stage1_validation/dev15_local_zotero_dual_source_audit.tsv` now map to `PAT_STAGE2_SCOPE_BINDING_REFRESH_V1`; `docs/audits/dev15_baseline_governance_membership_audit_2026-04-17.md` is now tracked as explicit `governance_support_change` | local-source recovery artifacts plus baseline governance membership audit |
| `AF008` | `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1` was labeled `active_mainline_pattern` despite failed compare-run activation gating | resolved conservatively | `status-overclaim` | downgraded to `candidate_historical_pattern` | `gt_authority_v2_variantaware/RUN_CONTEXT.md` records `run_activation_gate = fail`; feature activation report records `variant_aware_gt_authority_switch = missing` |
| `AF009` | Audit summary still claimed `23` tracked diff changes | resolved | `count-inconsistency` | corrected to `22` everywhere in the patch surfaces | `git diff --name-status 385b6e1..HEAD -- project docs src analysis data/results/20260417_385b6e1` returns 22 paths |

## Covered-By-Pattern
- `PAT_STAGE2_SCOPE_BINDING_REFRESH_V1`
  - current maintained Stage2 scope-binding repair
- `PAT_5GIF3D8W_SWEEP_AUTH_PRESERVATION_V1`
  - governed replay validation; `partially_restored`
- `PAT_QLYKLPKT_TWO_TABLE_PRESERVATION_V1`
  - governed replay validation; `partially_restored`
- `PAT_TABLE_SCOPE_ID_SYNTHESIS_V1`
  - new change-driven index row for the changed maintained script
- `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1`
  - change-driven governance row retained with conservative historical status because the cited compare run fails `run_activation_gate`

## Covered-But-Non-Pattern
- `data/results/20260417_385b6e1/10_qlyk_capability_restoration_v1`
  - classification: `supporting_run_without_pattern`
  - reason: no `RUN_CONTEXT.md`
- `project/2_ARCHITECTURE.md`
- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `docs/audits/dev15_baseline_governance_membership_audit_2026-04-17.md`
  - classification: `governance_support_change`
  - reason: authoritative governance linkage for current Stage5 diagnostic mode without pattern-activation proof

## Missing-Run-Surface
- `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix`
  - cited as validating replay evidence in the prior index row
  - not present on disk
- `data/results/20260417_385b6e1/10_qlyk_capability_restoration_v1`
  - run root exists, but no reproducibility-grade `RUN_CONTEXT.md`

## Status-Overclaim
- `PAT_WFDTQ4VX_SELECTOR_RESTORE_V1`
  - prior status: `active_mainline_pattern`
  - current evidence strength: `partial_indirect_activation`
- `PAT_UFXX9WXE_DOE_MAINLINE_RESTORE_V1`
  - prior status: `validated_replay_pattern`
  - current reproducibility status: `missing_run_surface`
- `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1`
  - prior status: `active_mainline_pattern`
  - current evidence strength: `partial_indirect_activation`
  - blocking fact: compare-side `RUN_CONTEXT.md` records `run_activation_gate = fail`

## Residual Follow-Up
- `PAT_TABLE_SCOPE_ID_SYNTHESIS_V1`
  - current governed replay does not prove repaired non-DOE expansion after synthesized scope IDs
- `PAT_BXCV5XWB_HELPER_DESCENDANT_GOVERNANCE_V1`
  - current baseline artifacts still mark `variant_aware_gt_authority_switch` as missing
- `PAT_5GIF3D8W_SWEEP_AUTH_PRESERVATION_V1`
  - validated replay remains `partially_restored`
- `PAT_QLYKLPKT_TWO_TABLE_PRESERVATION_V1`
  - validated replay remains `partially_restored`
