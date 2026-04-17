# DEV15 Baseline Governance Membership Audit (2026-04-17)

## Purpose

Audit the governance status of the existing DEV15 no-marker full live baseline rooted at:

- `data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live`

This audit is conservative and distinguishes:

- governed run-local artifacts
- governed audit artifacts
- external temporary artifacts
- central baseline-registry membership

## Existing governed run-local artifacts

The following baseline-relevant artifacts already existed inside the governed run lineage:

- `RUN_CONTEXT.md`
- `stage2_run_metadata_v1.json`
- `targeted_manifest.tsv`
- `semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
- `semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- `formulation_relation_records_v1.tsv`
- `resolved_relation_fields_v1.tsv`
- `final_formulation_table_v1.tsv`
- `final_output_decision_trace_v1.tsv`
- `final_output_summary_v1.md`
- `audit/identity_freeze_guardrail_v1/identity_freeze_report_v1.tsv`
- `audit/identity_freeze_guardrail_v1/identity_freeze_summary_v1.tsv`
- `audit/identity_freeze_guardrail_v1/identity_freeze_summary_v1.md`

These artifacts show that the run reached:

- maintained Stage2 live LLM completion
- maintained Stage3 relation materialization
- maintained Stage5 final-table materialization
- maintained identity-freeze execution

## Existing governed audit artifacts

Two daily baseline audit directories already existed under the active-run audit root:

- `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/audit/daily_baseline_v1/dev15_full_baseline_2026_04_17`
- `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/audit/daily_baseline_v1/dev15_full_pipeline_baseline_2026_04_17`

Both contain governed baseline-audit utility outputs such as:

- `audit_manifest.json`
- `baseline_input_fingerprint.json`
- `script_chain_snapshot.json`
- `feature_activation_snapshot.tsv`
- `stage_output_inventory.json`

However, both audit manifests point to:

- `source_run_dir = data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1`

not to:

- `data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live`

Therefore these daily baseline audit surfaces are governed, but they do not by themselves register the 2026-04-17 no-marker live run as a self-contained system-managed baseline object.

## Existing external-only artifacts before normalization

Before normalization, the `debug_identity` compare outputs existed only outside governed lineage paths:

- `/tmp/dev15_debug_identity_compare_20260417/final_table_vs_gt_counts.tsv`
- `/tmp/dev15_debug_identity_compare_20260417/final_table_vs_gt_summary.md`
- `/tmp/dev15_debug_identity_compare_20260417/final_table_vs_gt_counts_by_doi.tsv`
- `/tmp/dev15_debug_identity_compare_20260417/final_table_vs_gt_count_audit.md`
- `/tmp/dev15_debug_identity_compare_20260417/RUN_CONTEXT.md`

These outputs were reproducible and explicit, but not governed by repo run-lineage placement rules.

## Classification assessment

Best descriptive classification before normalization:

- development / diagnostic full-pipeline baseline

Best governed classification after normalization:

- `baseline_type = full_pipeline_baseline`
- `benchmark_validity = diagnostic_only`

The baseline is not benchmark-valid because:

- identity freeze failed
- the compare continuation is `debug_identity`

The baseline is not an exploratory baseline because:

- it traversed the maintained Stage2 live path through Stage5 final-table materialization
- it preserved maintained observability and run-context artifacts

The baseline is not a mere partial failed run because:

- it produced a governed final table
- it produced governed identity-freeze outputs
- it can support diagnostic-only final-table GT comparison

## System-management assessment

Before this normalization work, the baseline was:

- governed in substance at the run-artifact level
- partially governed at the audit level
- not fully system-managed as one explicit baseline object

Reasons:

- run-local Stage2 through identity-freeze artifacts were governed
- daily baseline audit artifacts existed but pointed to a different source run
- debug compare outputs were outside governed lineage paths
- no central `data/baselines` registration row or baseline manifest pointed to this run
- no run-local baseline registration artifact existed

## Normalization decision

The narrowest governed normalization is:

1. keep the existing run lineage as the authority root
2. materialize the `debug_identity` compare outputs under the run lineage
3. add a run-local baseline registration artifact
4. add a central `data/baselines` registry row plus manifest
5. keep benchmark validity explicit as `diagnostic_only`

## Conservative conclusion

This baseline was only partially managed before normalization.

After normalization, it is suitable to be treated as:

- the governed current DEV15 no-marker full live diagnostic baseline

It must not be referred to as:

- benchmark-valid baseline
- exploratory scratch run
- partial Stage2-only run
