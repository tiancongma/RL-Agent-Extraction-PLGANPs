# Repair Track Status (2026-04-14)

This audit records the current governed status of two repair tracks that future
agents must recover without re-deriving context from scattered run notes.

## Repair Track A - Stage5 benchmark legality / identity-freeze failure

### Facts

- Stage5 final-table generation is necessary but not sufficient for a
  benchmark-valid run.
- The full DEV15 lineage
  `data/results/20260401_5d9f4e6/09_dev15_count_validation`
  reached Stage5 final-table materialization.
- That same lineage then failed the mandatory identity-freeze gate, so the
  compare output and any downstream modeling-ready continuation from that
  lineage are not benchmark-valid.

### Failure classes

- row count drift
- identity reassignment
- unresolved scaffold binding

Primary evidence:

- `data/results/20260401_5d9f4e6/09_dev15_count_validation/RUN_CONTEXT.md`
- `data/results/20260412_8517d36/18_full_pipeline_benchmark_dev15_v1/audit/identity_freeze_guardrail_v1/identity_freeze_summary_v1.md`
- `data/results/20260412_8517d36/18_full_pipeline_benchmark_dev15_v1/audit/layer2_identity_scaffold_binding_v1/layer2_identity_scaffold_validation_v1.md`

### Current status

- discovered
- localized
- entered governed repair lineage
- partially repaired at scaffold and representation layers
- still the benchmark-validity gate for current full-pipeline runs

Scope note:

- Follow-on scaffold-binding and representation work helps explain and reduce
  the failure surface.
- Those follow-on repairs do not yet authorize any claim that the full issue is
  solved unless a lawful full-pipeline run passes the hard identity-freeze
  gate.

Relevant governed follow-on evidence:

- `docs/audits/workbook_provenance_audit_value_gt_annotation_workbook_representation_repaired_v4_with_pH_2026-04-13.md`

## Repair Track B - Functional-unit system not truly on the mainline after Stage2 decomposition

### Facts

- Stage2 decomposition introduced a real architecture failure:
  semantic signals could exist while governed deterministic functional units
  were not reliably taking control of execution.
- The intended contract remains:
  - LLM = semantic discovery and authorization
  - deterministic function units = execution
- Silent non-activation is not acceptable when semantic authorization is
  present.

Discovery evidence:

- `data/results/20260401_5d9f4e6/09_dev15_count_validation/analysis/feature_execution_ledger_v1.tsv`
- `data/results/20260401_5d9f4e6/09_dev15_count_validation/analysis/feature_execution_ledger_report_v1.md`

### What the audit showed

- some units existed in docs or code but were not provably active in the failed
  DEV15 lineage
- sequential optimization was active
- DOE and non-DOE table-row execution were not reliably on-path across that
  governed lineage

### What is repaired now

- The DOE execution path is restored on the mainline for the confirmed
  `UFXX9WXE` case:
  - run:
    `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix`
  - governed deterministic DOE rows emitted:
    `26`

Supporting evidence:

- `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix/RUN_CONTEXT.md`
- `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix/analysis/feature_activation_report_v1.tsv`
- `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix/semantic_to_widerow_adapter/compatibility_projection_summary_v1.json`
- `data/results/20260406_ced19d6/08_doe_trigger_path_audit/doe_trigger_diagnostics_v1.json`

### What remains partial

- Non-DOE table-row repair improved observability and corrected rule or
  unit-level issues for already-authorized cases.
- Broader DEV15 coverage is still blocked upstream because many papers do not
  arrive with `table_formulation_scopes`.
- The dominant remaining blocker is therefore upstream Stage2 extraction,
  selector, or evidence-handoff completeness rather than downstream execution
  willingness.

Primary evidence:

- `data/results/20260414_0011ee7/01_non_doe_table_row_repair_v1/analysis/root_cause_classification.md`
- `data/results/20260414_0011ee7/01_non_doe_table_row_repair_v1/analysis/table_row_repair_validation.md`
- `data/results/20260414_0011ee7/01_non_doe_table_row_repair_v1/analysis/summary.json`
- `data/results/20260414_0011ee7/01_non_doe_table_row_repair_v1/outputs/repaired_full_freeze/execution_ledger_v2.tsv`

### Current status

- architecture failure discovered
- DOE execution path repaired
- non-DOE execution partially repaired
- dominant remaining blocker moved upstream to Stage2 extraction or selector or
  evidence-handoff

## Intentionally deferred until a later lawful run

- Any claim that Repair Track A is fully solved
- Any claim that GT compare from the failed DEV15 Stage5 lineage is
  benchmark-valid
- Any claim that non-DOE table-row coverage is broadly restored across DEV15
  without upstream `table_formulation_scopes`
