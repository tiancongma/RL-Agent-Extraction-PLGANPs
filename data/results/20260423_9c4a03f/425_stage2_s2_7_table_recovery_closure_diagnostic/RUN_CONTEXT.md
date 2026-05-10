# RUN_CONTEXT

## 1. Run ID
`425_stage2_s2_7_table_recovery_closure_diagnostic`

## 1a. Run Path
- run_dir: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic`
- run_dir_kind: `v2_child_execution`
- run_selection_mode: `explicit_run_dir`
- bucket_dir: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f`

## 2. Run Type
`intermediate_diagnostic_run`

Benchmark reporting rule:
- This run is `diagnostic-only, not benchmark-valid final output`.
- It materializes the frozen `S2-7` completed Stage2 artifact only.
- It does not execute `Stage3`, `Stage4`, or `Stage5`.

## 3. Purpose
- Consume the `S2-6`-validated `S2-5` semantic-intermediate surface only.
- Invoke the maintained compatibility-projection logic without any new LLM call.
- Materialize the completed Stage2 artifact required by unchanged downstream `Stage3` consumers.
- Stop before `Stage3` execution.

## 4. Stage Boundary
- current_stage_boundary: `S2-7`
- boundary_class: `mainline_resume_boundary`
- authoritative_for_downstream: `yes`
- lawful_resume_boundary: `yes`
- resume_entrypoint: `src/stage3_relation/run_formulation_relation_artifacts_v1.py`
- stage_local_owner_script: `src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py`
- stage_local_owner_function_surface:
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py::run_projection`
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py::project_document`
- stop_boundary: `completed_stage2_artifact_written`
- next_lawful_step: `Stage3 relation materialization`

## 5. Inputs
- source_s2_6_run_dir: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/420_stage2_dev15_cleantext_current_s2_6_contract_diagnostic`
- source_validation_report: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/420_stage2_dev15_cleantext_current_s2_6_contract_diagnostic/analysis/stage2_semantic_authority_contract_report_v1.json`
- source_validation_status: `pass`
- source_s2_5_run_dir: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/419_stage2_dev15_cleantext_current_s2_5_parse_diagnostic`
- semantic_jsonl: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/419_stage2_dev15_cleantext_current_s2_5_parse_diagnostic/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
- authority_reattachment_sidecar: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/419_stage2_dev15_cleantext_current_s2_5_parse_diagnostic/semantic_stage2_objects/authority_reattachment_sidecar_v1.json`
- input_contract_note:
  - this runner requires a passing `S2-6` validation report before projection
  - the semantic JSONL path is resolved from the `S2-6` validation surface
  - authority reopen metadata is reattached from the deterministic S2-5 sidecar, not from LLM semantic content
  - this runner does not consume raw `S2-4b` responses
  - this runner does not perform semantic parsing or contract validation
  - this runner does not invoke any live model call

## 6. Exact Script Execution Order
1. `src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py`
2. `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py::run_projection`

## 7. Outputs
- completed Stage2 artifact directory:
  - `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter`
- completed Stage2 weak-label TSV:
  - `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- completed Stage2 weak-label JSONL:
  - `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.jsonl`
- compatibility trace TSV:
  - `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/compatibility_projection_trace_v1.tsv`
- compatibility summary JSON:
  - `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/compatibility_projection_summary_v1.json`
- function-unit activation report v2:
  - `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/analysis/feature_activation_report_v2.tsv`
- execution ledger v2:
  - `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/analysis/execution_ledger_v2.tsv`
- numbered DOE guard TSV:
  - `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/numbered_doe_regression_guard_v1.tsv`
- projection contract TSV:
  - `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/stage2_replacement_compatibility_projection_contract.tsv`
- run context:
  - `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/RUN_CONTEXT.md`
- machine-readable run metadata:
  - `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/stage2_s2_7_run_metadata_v1.json`

## 8. Stop Rule
- This run stops after `S2-7` compatibility projection artifacts are written.
- The following stages were not executed:
  - `Stage3`
  - `Stage4`
  - `Stage5`

## 9. Run Summary
- projected_document_count: `15`
- projected_row_count: `228`
- completed_stage2_artifact_status: `written`
- lawful_stage3_resume_boundary: `yes`

## 10. Function Unit Execution
- activation_report_v2:
  - `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/analysis/feature_activation_report_v2.tsv`
- execution_ledger_v2:
  - `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/analysis/execution_ledger_v2.tsv`
- tracked_fields:
  - `was_unit_considered`
  - `was_unit_authorized`
  - `was_unit_called`
  - `rows_emitted`
  - `rows_retained_after_projection`
  - `skip_reason`

## Boundary Governance

- `boundary_class`: `mainline_resume_boundary`
- `authoritative_for_downstream`: `yes`
- `lawful_resume_boundary`: `yes`
- `resume_entrypoint`: `src/stage3_relation/build_formulation_relation_artifacts_v1.py`
- `schema_contract`: `completed_stage2_compatibility_projection_v1`
- `upstream_authority_source`: `unknown`
- `replay_mode`: `none`
- `source_run_dir`: `not_applicable`
- `source_run_id`: `not_applicable`
- `source_files`: `[]`
- `benchmark_terminal`: `no`
- `boundary_evidence_path`: `data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`

## Feature Unit Activation

- `feature_activation_report_path`: `data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/analysis/feature_activation_report_v1.tsv`
- `required_feature_units`: `["numbered_doe_row_enumeration_priority", "numbered_doe_regression_guard", "table_first_evidence_binding", "s2_candidate_section_aware_split", "s2_candidate_noise_filtering", "s2_2_evidence_artifact_contract", "s2_2_design_success_split", "s2_2_prompt_preview_derived_from_evidence_artifact", "s2_2_evidence_priority_selection", "feature_unit_governance_layer", "run_context_feature_activation_integration"]`
- `missing_required_feature_units`: `["s2_candidate_section_aware_split", "s2_candidate_noise_filtering", "s2_2_evidence_artifact_contract", "s2_2_design_success_split", "s2_2_prompt_preview_derived_from_evidence_artifact", "s2_2_evidence_priority_selection"]`
- `run_activation_gate`: `fail`

| feature_id | expected_for_run | observed_activation | activation_status | activation_state | evidence_path |
|---|---|---|---|---|---|
| `numbered_doe_row_enumeration_priority` | `yes` | `active` | `active` | `active` | `data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv` |
| `numbered_doe_regression_guard` | `yes` | `active` | `active` | `active` | `data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/numbered_doe_regression_guard_v1.tsv` |
| `table_first_evidence_binding` | `yes` | `unclear` | `unclear` | `processing_error` | `data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv` |
| `stage2_input_evidence_packing` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_candidate_section_aware_split` | `yes` | `missing` | `missing` | `evidence_missing` | `` |
| `s2_candidate_table_isolation` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_2a_table_authority_ranking` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_2a_primary_table_guardrail` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_candidate_noise_filtering` | `yes` | `missing` | `missing` | `evidence_missing` | `` |
| `s2_2_evidence_artifact_contract` | `yes` | `missing` | `missing` | `evidence_missing` | `` |
| `s2_2_design_success_split` | `yes` | `missing` | `missing` | `evidence_missing` | `` |
| `s2_2_prompt_preview_derived_from_evidence_artifact` | `yes` | `missing` | `missing` | `evidence_missing` | `` |
| `s2_2_evidence_priority_selection` | `yes` | `missing` | `missing` | `evidence_missing` | `` |
| `s2_2_doe_overlay_selection` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_2_duplicate_table_suppression` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `stage2_json_sanitation_path1` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `stage2_legacy_fixparse_fallback_path2` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `variant_aware_gt_authority_switch` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `family_variant_retention_governance` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `feature_unit_governance_layer` | `yes` | `active` | `active` | `active` | `data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/analysis/feature_activation_report_v1.tsv` |
| `run_context_feature_activation_integration` | `yes` | `active` | `active` | `active` | `data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/RUN_CONTEXT.md` |
| `doi_level_gt_vs_pred_count_audit` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
