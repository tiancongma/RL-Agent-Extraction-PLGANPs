# Feature Governance Recovery Report v1

Diagnostic-only governance support artifact.

## Purpose

This package reconstructs the feature set that is supposed to exist around the active pipeline so future debugging can start from:

- what features were designed
- which ones are maintained and active by default
- which ones are conditional or off by default
- which ones are historical only
- which ones are only discussed or inconsistently represented
- which ones were expected vs actually active in a real run

It does not change Stage2, Stage3, Stage5, validators, or contracts.

## Artifacts in this package

- `docs/feature_governance/feature_master_inventory_v1.tsv`
- `docs/feature_governance/run_feature_contract_backfill_v1.tsv`
- `docs/feature_governance/feature_gap_watchlist_v1.tsv`
- `docs/feature_governance/feature_governance_summary_v1.json`

## Sources inspected

Governance and authority:

- `AGENTS.md`
- `README.md`
- `project/0_PROJECT_CHARTER.md`
- `project/1_REQUIREMENTS.md`
- `project/2_ARCHITECTURE.md`
- `project/PIPELINE_SCRIPT_MAP.md`
- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/FILE_NAMING_AND_VERSIONING.md`
- `project/ACTIVE_DATA_SOURCE_CONTRACT.md`
- `project/4_DECISIONS_LOG.md`

Feature governance and registries:

- `project/feature_units/FEATURE_UNIT_GOVERNANCE.md`
- `project/feature_units/feature_unit_registry.json`
- `project/feature_units/feature_intervention_matrix.tsv`
- `docs/maintained_script_surface.tsv`
- `docs/src_script_registry.tsv`
- `docs/tool_index.md`

Memory and historical design traces:

- `src/utils/mem_bootstrap_v1.py`
- `src/utils/query_mem_v1.py`
- `data/mem/v1/*`
- `docs/methods/llm_reuse_replay_policy_v1.md`
- `docs/methods/v7pilot_r3_fixparse_input_assembly_audit_2026-03-10.md`
- `docs/methods/v7pilot_r3_fixparse_block_packing_2026-03-10.md`
- `docs/methods/v7pilot_r3_fixparse_synthesis_method_block_2026-03-10.md`
- `docs/methods/l3h2rs2h_formulation_instance_regression_audit_2026-03-10.md`
- `docs/audits/sequential_optimization_fu_validation.md`

Maintained code paths:

- `src/stage2_sampling_labels/run_stage2_composite_v1.py`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
- `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py`
- `src/stage2_sampling_labels/table_row_expansion_v1.py`
- `src/stage2_sampling_labels/function_units/sequential_optimization_interpreter_v1.py`
- `src/stage5_benchmark/compare_final_table_to_gt_v1.py`
- `src/stage5_benchmark/enforce_identity_freeze_v1.py`
- `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py`
- `src/stage5_benchmark/build_layer2_risk_assessment_v1.py`
- `src/utils/build_feature_activation_report_v1.py`
- `src/utils/update_run_context_with_feature_activation_v1.py`
- `src/utils/active_data_source.py`

Representative runs and artifacts:

- `data/results/20260407_ab12cd3/12_semantic_authority_recovery_qlyk_probe_example/RUN_CONTEXT.md`
- `data/results/20260407_ab12cd3/12_semantic_authority_recovery_qlyk_probe_example/analysis/stage2_prompt_preview_v1.tsv`
- `data/results/20260407_ab12cd3/12_semantic_authority_recovery_qlyk_probe_example/analysis/feature_activation_report_v1.tsv`
- `data/results/20260407_ab12cd3/12_semantic_authority_recovery_qlyk_probe_example/semantic_to_widerow_adapter/compatibility_projection_summary_v1.json`
- `data/results/20260407_ab12cd3/17_qlyk_seqopt_patch_validation/RUN_CONTEXT.md`
- `data/results/20260407_ab12cd3/17_qlyk_seqopt_patch_validation/semantic_to_widerow_adapter/compatibility_projection_summary_v1.json`
- `data/results/run_20260331_1510_03e5d25_subset5_stage2_summary_regression_v1/RUN_CONTEXT.md`

## Memory queries and recovery commands used

- `python src/utils/mem_bootstrap_v1.py --query "QLYKLPKT Stage2 input construction ordered input optimization summary mode prompt layout class raw_prefix_then_table_excerpts activation audit"`
- `python src/utils/query_mem_v1.py --query "feature activation stage2 input evidence packing summary table sequential optimization marker readiness replay boundary risk metadata audit ready export compatibility projection" --limit 20`
- `python src/utils/query_mem_v1.py --query "table-first block packing synthesis-first summary first column evidence pack stage2 prompt ordering discussed feature" --limit 30`

## Recovery method

1. Read the governance layer first and lock authoritative pipeline and run-source rules.
2. Query governed memory to recover prior decisions, solved regressions, and older feature intent that might not be obvious from current code alone.
3. Read the existing feature-unit governance surfaces to see what is already formally registered.
4. Compare maintained code, maintained script surfaces, and historical method notes to recover active maintained features, active conditional features, historical but easy-to-confuse features, and review-needed items.
5. Backfill representative real runs so expected-vs-actual status is artifact-backed.

## Clearly active-maintained features

- governed composite Stage2 entrypoint
- llm-first composite semantic authority
- semantic authority contract validator
- completed Stage2 as the only lawful Stage3 resume boundary
- compatibility projection completion
- default raw-prefix then table-excerpts Stage2 layout
- partial selection and inheritance marker preservation
- execution-ready-only downstream handshake
- feature activation report
- RUN_CONTEXT feature activation section
- boundary governance metadata
- active data-source authority resolution
- identity freeze hard gate
- family and variant retention governance

## Active conditional features

- live-v2 raw-response rehydration
- ordered input evidence packing
- prompt preview observability
- summary table mode
- summary first-column enhancement
- non-DOE table row expansion
- sequential optimization interpreter
- DOE expansion within LLM-declared scope
- JSON sanitation path 1
- audit-ready export
- Layer2 paper risk assessment
- compare-time DOI-level count audit
- compare-time variant-aware GT authority switch
- memory bootstrap for complex debugging
- deterministic semantic emitter fallback mode

## Historical or non-mainline features recovered so they are not forgotten

- Stage2.5 shadow evidence-pack builder
- legacy table-first block packing in the deprecated wide-row extractor
- legacy synthesis-method and materials-procurement block ordering
- legacy table-heavy row-enumeration prompt hint
- legacy fixparse fallback path 2

## Features previously discussed or implemented but not safely enforced today

- summary mode and summary-first-column enhancement are in maintained code and governed run history, but not in the current feature-unit registry or observer layer
- replay runs can hide original live prompt assembly unless the parent live run is explicitly audited
- a compare-only GT authority feature is currently treated as required in Stage2-only runs by the existing feature-activation gate
- auxiliary docs still contain at least one stale Stage2 authority statement

## Initial run-level backfill conclusion

- `data/results/20260407_ab12cd3/12_semantic_authority_recovery_qlyk_probe_example`
  - maintained Stage2 path was used
  - ordered packing was not active
  - prompt preview was active
  - table mode was full, not summary
  - sequential optimization interpreter was active
  - run gate failed for a compare-only feature requirement unrelated to the Stage2 audit
- `data/results/20260407_ab12cd3/17_qlyk_seqopt_patch_validation`
  - replay rehydration was active
  - replay did not provide its own prompt-preview artifact
  - parent live run remains the authoritative prompt audit surface
- `data/results/run_20260331_1510_03e5d25_subset5_stage2_summary_regression_v1`
  - summary mode existed in the maintained path and was explicitly exercised

## Required gap report

See `docs/feature_governance/feature_gap_watchlist_v1.tsv`.

## Adjudicated Feature Model (Post-Recovery)

The governance-support system now distinguishes three layers:

- input contract layer
- semantic feature-unit layer
- downstream deterministic layer

Input-construction features belong to the input contract layer.

- they are not triggered by the LLM
- they must be satisfied before semantic processing begins
- they are governed outside the feature-unit system

Feature-unit system scope is limited to semantic-triggered behavior and related governed execution units. It is not the authority surface for Stage2 input-construction policy.

Fallback modes such as raw-prefix prompt assembly remain documented, but they are explicitly not default-safe. They are retained for compatibility, audit traceability, and controlled fallback diagnosis rather than as the adjudicated default input contract.
