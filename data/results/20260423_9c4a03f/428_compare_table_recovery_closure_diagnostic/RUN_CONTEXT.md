# RUN_CONTEXT

## 1. Run Type

- `diagnostic_baseline_run`

## 2. Purpose

- Compare the completed Stage 5 final table against the authoritative frozen Layer1 GT counts TSV for the declared scope.

## 3. Source Authority Resolution

- source_resolution: `active_run_pointer`
- source_run_id: `20260423_9c4a03f`
- source_run_dir: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f`
- active_run_pointer_path: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/ACTIVE_RUN.json`
- scope_manifest_tsv: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`
- final_table_tsv: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/427_stage5_table_recovery_closure_diagnostic/final_formulation_table_v1.tsv`
- layer1_gt_counts_tsv: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/cleaned/gt_authority/v1/dev15_layer1_gt_counts.tsv`

## 4. Exact Script Execution Order

1. Run `src/stage5_benchmark/compare_final_table_to_gt_v1.py` on the completed Stage 5 final table.
2. Refresh `RUN_CONTEXT.md` via `src/utils/update_run_context_with_feature_activation_v1.py` so feature activation lineage is recorded in the compare run.

## 5. Outputs

- `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/428_compare_table_recovery_closure_diagnostic/final_table_vs_gt_counts.tsv`
- `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/428_compare_table_recovery_closure_diagnostic/final_table_vs_gt_summary.md`
- `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/428_compare_table_recovery_closure_diagnostic/final_table_vs_gt_counts_by_doi.tsv`
- `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/428_compare_table_recovery_closure_diagnostic/final_table_vs_gt_count_audit.md`
- `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/428_compare_table_recovery_closure_diagnostic/RUN_CONTEXT.md`

## 6. Benchmark Status

- compare_mode: `diagnostic`
- benchmark_mode: `disabled`
- benchmark_valid: `no`
- `diagnostic-only, not benchmark-valid final output`
- Reason: current phase is diagnostic development mode.

## 7. Reproduction Metadata

- generated_at: `2026-05-10T07:29:15`
- scope_name: `dev15_diagnostic_table_recovery_closure`
- final_table_rows: `200`
- gt_rows: `202`
- matched_papers: `13`
- mismatched_papers: `2`

## Boundary Governance

- `boundary_class`: `benchmark_terminal_boundary`
- `authoritative_for_downstream`: `yes`
- `lawful_resume_boundary`: `no`
- `resume_entrypoint`: `not_applicable`
- `schema_contract`: `stage5_final_table_vs_gt_comparison_v1`
- `upstream_authority_source`: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f`
- `replay_mode`: `none`
- `source_run_dir`: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f`
- `source_run_id`: `20260423_9c4a03f`
- `source_files`: `["/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv", "/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv", "/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/427_stage5_table_recovery_closure_diagnostic/final_formulation_table_v1.tsv"]`
- `benchmark_terminal`: `yes`
- `boundary_evidence_path`: `data/results/20260423_9c4a03f/428_compare_table_recovery_closure_diagnostic/final_table_vs_gt_counts.tsv`

## Feature Unit Activation

- `feature_activation_report_path`: `data/results/20260423_9c4a03f/428_compare_table_recovery_closure_diagnostic/analysis/feature_activation_report_v1.tsv`
- `required_feature_units`: `["variant_aware_gt_authority_switch", "feature_unit_governance_layer", "run_context_feature_activation_integration", "doi_level_gt_vs_pred_count_audit"]`
- `missing_required_feature_units`: `[]`
- `run_activation_gate`: `pass`

| feature_id | expected_for_run | observed_activation | activation_status | activation_state | evidence_path |
|---|---|---|---|---|---|
| `numbered_doe_row_enumeration_priority` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `numbered_doe_regression_guard` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `table_first_evidence_binding` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `stage2_input_evidence_packing` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_candidate_section_aware_split` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_candidate_table_isolation` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_2a_table_authority_ranking` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_2a_primary_table_guardrail` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_candidate_noise_filtering` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_2_evidence_artifact_contract` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_2_design_success_split` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_2_prompt_preview_derived_from_evidence_artifact` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_2_evidence_priority_selection` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_2_doe_overlay_selection` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_2_duplicate_table_suppression` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `stage2_json_sanitation_path1` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `stage2_legacy_fixparse_fallback_path2` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `variant_aware_gt_authority_switch` | `yes` | `active` | `active` | `active` | `data/results/20260423_9c4a03f/428_compare_table_recovery_closure_diagnostic/final_table_vs_gt_counts_by_doi.tsv` |
| `family_variant_retention_governance` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `feature_unit_governance_layer` | `yes` | `active` | `active` | `active` | `data/results/20260423_9c4a03f/428_compare_table_recovery_closure_diagnostic/analysis/feature_activation_report_v1.tsv` |
| `run_context_feature_activation_integration` | `yes` | `active` | `active` | `active` | `data/results/20260423_9c4a03f/428_compare_table_recovery_closure_diagnostic/RUN_CONTEXT.md` |
| `doi_level_gt_vs_pred_count_audit` | `yes` | `active` | `active` | `active` | `data/results/20260423_9c4a03f/428_compare_table_recovery_closure_diagnostic/final_table_vs_gt_counts_by_doi.tsv` |
