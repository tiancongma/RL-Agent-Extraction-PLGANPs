# Repair Index Acceptance Patch Report v1

## Summary
- Scope: final minimal patch to close the remaining repair-index re-acceptance blockers without redesigning the system.
- Baseline anchor: `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1`
- Recomputed tracked post-baseline diff count: `22`

## Finding Log

| finding_id | subject | old_status | new_status | reason | evidence |
|---|---|---|---|---|---|
| `AF001` | `data/results/20260417_385b6e1/10_qlyk_capability_restoration_v1` | `untracked_in_index` | `supporting_run_without_pattern` | Run surface exists, but no `RUN_CONTEXT.md`; cannot support promotion. | `find .../10_qlyk_capability_restoration_v1 -name RUN_CONTEXT.md` |
| `AF002` | `src/stage5_benchmark/enforce_identity_freeze_v1.py`; `src/stage5_benchmark/compare_final_table_to_gt_v1.py` | `untracked_in_index` | `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1 candidate_historical_pattern` | Maintained entrypoint changes are linked to governed diagnostic compare evidence and authoritative governance files, but the compare-side activation gate fails. | `11_5gif.../gt_authority_v2_variantaware/RUN_CONTEXT.md`; Stage5 diffs |
| `AF003` | `src/stage2_sampling_labels/table_row_expansion_v1.py` | `untracked_in_index` | `PAT_TABLE_SCOPE_ID_SYNTHESIS_V1 candidate_historical_pattern` | Changed maintained script is now tracked, but repaired downstream effect is not proven by a governed run. | post-baseline diff; `12_qlyk.../compatibility_projection_summary_v1.json` |
| `AF004` | `PAT_WFDTQ4VX_SELECTOR_RESTORE_V1` | `active_mainline_pattern` | `candidate_historical_pattern` | Current artifacts preserve anchor intent, but explicit governed activation is not proven. | baseline `09/RUN_CONTEXT.md`; baseline feature activation report |
| `AF005` | `PAT_UFXX9WXE_DOE_MAINLINE_RESTORE_V1` | `validated_replay_pattern` | `candidate_historical_pattern` | Cited validating run surface is missing on disk. | `find data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix` |
| `AF006` | `project/2_ARCHITECTURE.md`; `project/ACTIVE_PIPELINE_FLOW.md`; `project/ACTIVE_PIPELINE_RUNBOOK.md` | `audit_only_unlinked` | `linked to PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1` | Governance-file changes are now represented in repair-index linkage. | repair-index row linkage fields |
| `AF007` | `analysis/stage1_validation/dev15_clean_text_completeness.tsv`; `analysis/stage1_validation/dev15_local_source_mapping_validation.md`; `analysis/stage1_validation/dev15_local_zotero_dual_source_audit.tsv` | `uncovered_tracked_change` | `linked to PAT_STAGE2_SCOPE_BINDING_REFRESH_V1` | These local-source recovery artifacts directly support the maintained Stage1-to-Stage2 scope-binding repair and now have explicit audit coverage. | local-source recovery artifacts |
| `AF008` | `docs/audits/dev15_baseline_governance_membership_audit_2026-04-17.md` | `uncovered_tracked_change` | `explicit governance_support_change` | The file is materially relevant governance support for baseline classification, but it does not itself prove activation of a maintained repair unit. | baseline governance membership audit |
| `AF009` | `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1` | `active_mainline_pattern` | `candidate_historical_pattern` | The cited compare evidence run is reproducible but records `run_activation_gate = fail` and `variant_aware_gt_authority_switch = missing`, so mainline activation is overclaimed. | `gt_authority_v2_variantaware/RUN_CONTEXT.md`; compare feature activation report |
| `AF010` | audit tracked-diff summary | `23` | `22` | The actual recomputed `git diff --name-status 385b6e1..HEAD -- project docs src analysis data/results/20260417_385b6e1` inventory contains 22 tracked paths. | recomputed diff inventory |

## Status Transitions

### Downgraded
- `PAT_WFDTQ4VX_SELECTOR_RESTORE_V1`
  - `active_mainline_pattern -> candidate_historical_pattern`
- `PAT_UFXX9WXE_DOE_MAINLINE_RESTORE_V1`
  - `validated_replay_pattern -> candidate_historical_pattern`
- `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1`
  - `active_mainline_pattern -> candidate_historical_pattern`

### Added
- `PAT_TABLE_SCOPE_ID_SYNTHESIS_V1`
  - new change-driven tracking row for `table_row_expansion_v1.py`
- `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1`
  - new change-driven governance row for maintained Stage5 and governance-file deltas

## Final Minimal Patch Changes
- Coverage closed for `analysis/stage1_validation/dev15_clean_text_completeness.tsv`, `analysis/stage1_validation/dev15_local_source_mapping_validation.md`, and `analysis/stage1_validation/dev15_local_zotero_dual_source_audit.tsv` by linking them to `PAT_STAGE2_SCOPE_BINDING_REFRESH_V1`.
- Coverage closed for `docs/audits/dev15_baseline_governance_membership_audit_2026-04-17.md` as explicit non-pattern `governance_support_change`.
- `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1` was downgraded from `active_mainline_pattern` to `candidate_historical_pattern`.
- The tracked post-baseline diff count was corrected from `23` to `22` across the patched audit surfaces.

## Residual Non-Blocking Follow-Up
- `PAT_TABLE_SCOPE_ID_SYNTHESIS_V1`
  - no governed replay yet proves repaired non-DOE expansion after synthesized scope IDs
- `PAT_BXCV5XWB_HELPER_DESCENDANT_GOVERNANCE_V1`
  - current run-local compare surfaces still lack `variant_aware_gt_authority_switch`
- `PAT_5GIF3D8W_SWEEP_AUTH_PRESERVATION_V1`
  - replay-validated but still `partially_restored`
- `PAT_QLYKLPKT_TWO_TABLE_PRESERVATION_V1`
  - replay-validated but still `partially_restored`

## Final Self-Check
1. All 22 tracked post-baseline changes are now covered by a pattern or explicit non-pattern status: `yes`.
2. Any `active_mainline_pattern` still relying on a run with `run_activation_gate = fail`: `no`.
3. Tracked diff count now consistently `22` in the patched audit surfaces: `yes`.
