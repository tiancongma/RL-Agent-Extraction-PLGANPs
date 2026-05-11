# Feature Execution Ledger System Report v1

Diagnostic-only governance-support artifact.

## What This Adds

- global run-evaluable schema: `docs/feature_governance/feature_applicability_schema_v1.tsv`
- per-run ledger rows for every recovered feature
- explicit upstream processing trace for resume and replay runs
- run-local markdown reports that answer expected-vs-actual before semantic analysis

## Backfilled Run Types

- `21_qlyk_table_selection_fix_validation_v3` -> `stage2_live` with `live_llm`; mismatches=`4`

## State Model

- `expected_active`
- `active_observed`
- `active_inferred_from_upstream`
- `not_applicable_for_run_scope`
- `not_reachable_due_to_resume_boundary`
- `expected_but_not_observed`
- `intentionally_disabled`
- `replay_hidden`
- `unknown_needs_review`

## Use Before Debugging

1. Open the run-local `analysis/feature_execution_ledger_v1.tsv`.
2. Read `analysis/feature_upstream_processing_trace_v1.json`.
3. Read `analysis/feature_execution_ledger_report_v1.md`.
4. Only then inspect semantic extraction or GT comparison details.
