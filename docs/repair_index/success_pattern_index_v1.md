# Success Pattern Index v1

## Status Key
- `candidate_historical_pattern`: historically evidenced or governance-backed, but current governed activation or replay validation is incomplete
- `validated_replay_pattern`: repaired behavior is shown in a governed replay validation, but not yet integrated as a current baseline capability
- `active_mainline_pattern`: maintained entrypoint wiring and governed run activation are both explicit under the current architecture
- `rejected_or_obsolete_pattern`: do not reuse or promote as a current repair strategy

## Evidence Fields
- `covered_change_ids`: audit rows or change-inventory rows linked to the pattern
- `evidence_run_path`: governed run root used for current evidence, if any
- `evidence_run_context_path`: exact `RUN_CONTEXT.md` used for reproducibility review
- `evidence_feature_activation_path`: run-local feature activation evidence, if any
- `evidence_boundary_governance_path`: governed surface proving the claimed boundary
- `activation_evidence_strength`: `explicit_governed_activation`, `partial_indirect_activation`, `code_presence_only`, or `missing`
- `reproducibility_status`: `reproducible_governed_run`, `replay_only_governed_run`, `historical_unverified`, or `missing_run_surface`
- `pattern_scope_type`: `paper_specific_repair`, `cross_paper_feature_repair`, `benchmark_governance_repair`, or `audit_or_contract_support`
- `linked_governance_changes`: authoritative governance files or support contracts linked to the row
- `acceptance_audit_status`: `passes_current_audit`, `downgraded_due_to_missing_evidence`, `incomplete_coverage`, or `requires_followup`

## Summary Table

| pattern_id | adoption_status | pattern_scope_type | acceptance_audit_status | covered_change_ids | evidence_run_path | activation_evidence_strength | reproducibility_status |
|---|---|---|---|---|---|---|---|
| `PAT_STAGE2_SCOPE_BINDING_REFRESH_V1` | `active_mainline_pattern` | `cross_paper_feature_repair` | `passes_current_audit` | `AUD007;AUD008;AUD015;AUD017;AUD025;AUD026;AUD027` | `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2` | `explicit_governed_activation` | `replay_only_governed_run` |
| `PAT_WFDTQ4VX_SELECTOR_RESTORE_V1` | `candidate_historical_pattern` | `paper_specific_repair` | `downgraded_due_to_missing_evidence` | `AUD009;AUD021` | `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1` | `partial_indirect_activation` | `reproducible_governed_run` |
| `PAT_UFXX9WXE_DOE_MAINLINE_RESTORE_V1` | `candidate_historical_pattern` | `paper_specific_repair` | `downgraded_due_to_missing_evidence` | `AUD018` | `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix` | `missing` | `missing_run_surface` |
| `PAT_5GIF3D8W_SWEEP_AUTH_PRESERVATION_V1` | `validated_replay_pattern` | `paper_specific_repair` | `passes_current_audit` | `AUD015;AUD019` | `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2` | `explicit_governed_activation` | `replay_only_governed_run` |
| `PAT_QLYKLPKT_TWO_TABLE_PRESERVATION_V1` | `validated_replay_pattern` | `paper_specific_repair` | `passes_current_audit` | `AUD014;AUD016;AUD017;AUD020` | `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1` | `explicit_governed_activation` | `replay_only_governed_run` |
| `PAT_TABLE_SCOPE_ID_SYNTHESIS_V1` | `candidate_historical_pattern` | `cross_paper_feature_repair` | `requires_followup` | `AUD010;AUD017` | `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1` | `code_presence_only` | `historical_unverified` |
| `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1` | `candidate_historical_pattern` | `benchmark_governance_repair` | `downgraded_due_to_missing_evidence` | `AUD003;AUD004;AUD005;AUD011;AUD012` | `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2/gt_authority_v2_variantaware` | `partial_indirect_activation` | `reproducible_governed_run` |
| `PAT_BXCV5XWB_HELPER_DESCENDANT_GOVERNANCE_V1` | `candidate_historical_pattern` | `paper_specific_repair` | `requires_followup` | `AUD022;AUD024` | `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1` | `partial_indirect_activation` | `reproducible_governed_run` |
| `PAT_SORTED_CSV_FIRST4_MULTI_TABLE_FALLBACK_V1` | `rejected_or_obsolete_pattern` | `paper_specific_repair` | `passes_current_audit` | `AUD017;AUD020` | `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1` | `explicit_governed_activation` | `replay_only_governed_run` |

## Pattern Details

### `PAT_STAGE2_SCOPE_BINDING_REFRESH_V1`
- Title: Refresh Stage2 `targeted_manifest.tsv` from maintained text and table authority before execution.
- Covered changes: `src/stage1_cleaning/pdf2clean.py`, `src/stage2_sampling_labels/run_stage2_composite_v1.py`, `analysis/dev15_clean_text_completeness.tsv`, `analysis/dev15_local_source_mapping_validation.md`, `analysis/dev15_local_zotero_dual_source_audit.tsv`, and governed replay runs that executed after the overlay refresh.
- Evidence run: `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2`
- Evidence run context: `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2/RUN_CONTEXT.md`
- Feature activation: `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2/analysis/feature_activation_report_v1.tsv`
- Boundary governance: `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2/RUN_CONTEXT.md`
- Linked governance changes: none
- Notes: Current post-baseline governed replays depend on this maintained Stage2 scope-binding overlay. The local-source support artifacts record `15/15` DEV15 clean-text coverage, additive local path remapping, and the DEV15 single-family HTML/PDF availability audit used to validate the repaired Stage1-to-Stage2 authority handoff.

### `PAT_WFDTQ4VX_SELECTOR_RESTORE_V1`
- Title: Preserve WFDTQ4VX table-recovery behavior through the maintained selector path.
- Covered changes: `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
- Evidence run: `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1`
- Evidence run context: `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/RUN_CONTEXT.md`
- Feature activation: `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/analysis/feature_activation_report_v1.tsv`
- Boundary governance: `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/RUN_CONTEXT.md`
- Linked governance changes: none
- Notes: Downgraded because the run purpose says the capability was preserved, but no explicit run-local feature activation proves WFDTQ4VX-specific activation under the current acceptance rule.

### `PAT_UFXX9WXE_DOE_MAINLINE_RESTORE_V1`
- Title: Preserve DOE authority-table handoff and lawful DOE execution for UFXX9WXE-class papers.
- Covered changes: baseline contrast only; no verified post-baseline validating run surface exists.
- Evidence run: `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix`
- Evidence run context: missing
- Feature activation: missing
- Boundary governance: missing
- Linked governance changes: none
- Notes: Downgraded because the cited validating run path does not exist. The historical decision-log claim remains useful context, but replay validation is unproven on disk.

### `PAT_5GIF3D8W_SWEEP_AUTH_PRESERVATION_V1`
- Title: Preserve blank-tolerant sweep-bearing authority and the optimized-table floor for 5GIF3D8W.
- Covered changes: governed replay validation with explicit before/after comparison against baseline `09`.
- Evidence run: `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2`
- Evidence run context: `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2/RUN_CONTEXT.md`
- Feature activation: `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2/analysis/feature_activation_report_v1.tsv`
- Boundary governance: `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2/RUN_CONTEXT.md`
- Linked governance changes: none
- Notes: Replay validation satisfies the governed-run checks, but the run truth remains `partially_restored` with `8` final rows instead of the historical `24` or GT `26`.

### `PAT_QLYKLPKT_TWO_TABLE_PRESERVATION_V1`
- Title: Preserve both true sequential-optimization tables at S2-2b and keep sequential execution available downstream.
- Covered changes: governed prompt checkpoint, governed replay validation, and the non-compliant supporting live-attempt directory.
- Evidence run: `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1`
- Evidence run context: `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1/RUN_CONTEXT.md`
- Feature activation: `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1/analysis/feature_activation_report_v1.tsv`
- Boundary governance: `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1/RUN_CONTEXT.md`
- Linked governance changes: none
- Notes: Replay validation satisfies the contract and explicitly records preservation of both real tables, but the run remains `partially_restored` because Stage5 still emitted `10` rows instead of the expected `7`.

### `PAT_TABLE_SCOPE_ID_SYNTHESIS_V1`
- Title: Synthesize stable table formulation scope IDs before non-DOE row expansion.
- Covered changes: `src/stage2_sampling_labels/table_row_expansion_v1.py`
- Evidence run: `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1`
- Evidence run context: `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1/RUN_CONTEXT.md`
- Feature activation: none specific to this scope-id synthesis change
- Boundary governance: `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1/RUN_CONTEXT.md`
- Linked governance changes: none
- Notes: Added to make the changed maintained script visible in the index. Current governed replay still records `table_row_expansion_v1` with `skip_reason = missing_table_formulation_scopes`, so repaired downstream effect is not yet proven.

### `PAT_STAGE5_DIAGNOSTIC_BASELINE_GOVERNANCE_V1`
- Title: Align Stage5 identity freeze and compare entrypoints with diagnostic-development governance.
- Covered changes: `project/2_ARCHITECTURE.md`, `project/ACTIVE_PIPELINE_FLOW.md`, `project/ACTIVE_PIPELINE_RUNBOOK.md`, `src/stage5_benchmark/enforce_identity_freeze_v1.py`, and `src/stage5_benchmark/compare_final_table_to_gt_v1.py`
- Evidence run: `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2/gt_authority_v2_variantaware`
- Evidence run context: `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2/gt_authority_v2_variantaware/RUN_CONTEXT.md`
- Feature activation: `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2/gt_authority_v2_variantaware/analysis/feature_activation_report_v1.tsv`
- Boundary governance: `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2/gt_authority_v2_variantaware/RUN_CONTEXT.md`
- Linked governance changes: `project/2_ARCHITECTURE.md`; `project/ACTIVE_PIPELINE_FLOW.md`; `project/ACTIVE_PIPELINE_RUNBOOK.md`
- Notes: This row tracks the maintained Stage5 diagnostic-baseline mode as a change-driven governance repair, but it is not a supported active-mainline pattern. The cited compare run is reproducible and diagnostic-only, yet its `RUN_CONTEXT.md` records `run_activation_gate = fail` and `missing_required_feature_units = ["variant_aware_gt_authority_switch"]`, so the row is conservatively retained as historical governance evidence rather than current mainline activation.

### `PAT_BXCV5XWB_HELPER_DESCENDANT_GOVERNANCE_V1`
- Title: Filter parent-linked helper or control descendants at Stage5 without reopening Stage2.
- Covered changes: baseline compare and feature activation ambiguity only.
- Evidence run: `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1`
- Evidence run context: `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/RUN_CONTEXT.md`
- Feature activation: `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/analysis/feature_activation_report_v1.tsv`
- Boundary governance: `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/RUN_CONTEXT.md`
- Linked governance changes: none
- Notes: Remains historical because current run-local compare surfaces still mark `variant_aware_gt_authority_switch` as missing.

### `PAT_SORTED_CSV_FIRST4_MULTI_TABLE_FALLBACK_V1`
- Title: Do not rely on unconditional `sorted_csv_first_4` fallback for multi-table optimization papers.
- Covered changes: QLYK replay validation that explicitly states the repaired capability no longer falls back to `sorted_csv_first_4`.
- Evidence run: `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1`
- Evidence run context: `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1/RUN_CONTEXT.md`
- Feature activation: `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1/analysis/feature_activation_report_v1.tsv`
- Boundary governance: `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1/RUN_CONTEXT.md`
- Linked governance changes: none
- Notes: This row remains a negative guard so the legacy fallback is not reintroduced as a repair strategy.
