# Docs Cleanup Report v1

## Summary

This pass organized the `docs/` tree into clearer category buckets without
moving anything out of `project/`, touching code, or changing `data/results/`.

The tree already had several coherent specialized areas:

- `docs/methods/`
- `docs/audits/`
- `docs/working/`
- `docs/design/`
- `docs/parking_lot/`
- `docs/snapshots/`
- `docs/repo_integrity_checks/`
- `docs/benchmarks/`

The main remaining noise before cleanup was a cluster of top-level audit,
working, and navigation files.

## Before And After

Before cleanup:

- `docs/` files scanned before changes: `111`
- top-level files outside specialized subdirectories: `23`

After cleanup:

- files moved into normalized buckets: `13`
- new directories created: `1`
  - `docs/indexes/`
- top-level files outside specialized subdirectories after cleanup: `11`
- new centralized navigation artifacts:
  - `docs/indexes/DOCS_INDEX.md`
  - `docs/indexes/docs_moves_index.tsv`

## Counts By Category

Pre-cleanup category counts used for the inventory:

- `governance_support`: `5`
- `method`: `45`
- `audit`: `28`
- `working_note`: `4`
- `design_draft`: `6`
- `parking_lot`: `1`
- `index_navigation`: `2`
- `registry_reference`: `4`
- `snapshot_report`: `10`
- `other`: `6`

## Files Moved

- `docs/project_moved_files_index.md` -> `docs/indexes/project_moved_files_index.md`
- `docs/current_engineering_runs_backfill_plan.md` -> `docs/working/current_engineering_runs_backfill_plan.md`
- `docs/script_disposition_candidates.md` -> `docs/working/script_disposition_candidates.md`
- `docs/dev15_current_benchmark_audit_run_20260313_0950_f4912f3.md` -> `docs/audits/dev15_current_benchmark_audit_run_20260313_0950_f4912f3.md`
- `docs/dev15_review_workbook_v1.md` -> `docs/methods/dev15_review_workbook_v1.md`
- `docs/cleanup_wave1_post_state_summary.md` -> `docs/audits/cleanup_wave1_post_state_summary.md`
- `docs/cleanup_wave1_run_actions.md` -> `docs/audits/cleanup_wave1_run_actions.md`
- `docs/cleanup_wave1_script_actions.md` -> `docs/audits/cleanup_wave1_script_actions.md`
- `docs/cleanup_wave2_code_moves.md` -> `docs/audits/cleanup_wave2_code_moves.md`
- `docs/cleanup_wave2_run_moves.md` -> `docs/audits/cleanup_wave2_run_moves.md`
- `docs/cleanup_wave2_script_state_summary.md` -> `docs/audits/cleanup_wave2_script_state_summary.md`
- `docs/historical_script_triage_v1.tsv` -> `docs/audits/historical_script_triage_v1.tsv`
- `docs/historical_script_triage_priority.tsv` -> `docs/audits/historical_script_triage_priority.tsv`

## Files Left In Place Intentionally

- `docs/tool_index.md`
  - left at top level because it has direct references from project docs,
    registry docs, and support scripts
- `docs/maintained_script_surface.tsv`
- `docs/src_script_registry.tsv`
- `docs/src_script_compliance_report.tsv`
- `docs/run_directory_compliance_report.tsv`
  - left at top level because these are high-reference registry/reference
    surfaces already used directly by governance docs
- `docs/run_spec_template.md`
- `docs/mdec084_writer_contract_v1.md`
  - left at top level because they are actively referenced governance-support
  docs and moving them would create unnecessary reference churn
- `docs/governance/SCRIPT_GOVERNANCE_POLICY.md`
- `docs/governance/mdec084_sync_note.md`
- `docs/governance/project_governance_cleanup_report_v1.md`
  - moved into `docs/governance/` during the top-level finish pass because
    they had low enough reference pressure to relocate cleanly
- specialized folders such as `docs/methods/`, `docs/audits/`,
  `docs/governance/`, `docs/repo_integrity_checks/`,
  `docs/ee_coverage_rl/`, and `docs/snapshots/`
  - left in place because they are already coherent category buckets

## Remaining Noisy Or Ambiguous Areas

- `docs/ee_coverage_rl/`
  - coherent as a specialized branch-diagnostics area, but still historically
    noisy and mixed in purpose
- top-level governance-support docs
  - `docs/run_spec_template.md`
  - `docs/mdec084_writer_contract_v1.md`
  - these remain top-level because active `project/` governance docs still
    reference those exact paths

## Confirmation

No authoritative project contracts were moved out of `project/`.

## Top-level Finish Pass

This finishing pass reduced the remaining top-level noise without reopening the
broader `docs/` reorganization.

Files moved in this pass:

- `docs/mdec084_sync_note.md` -> `docs/governance/mdec084_sync_note.md`
- `docs/project_governance_cleanup_report_v1.md` ->
  `docs/governance/project_governance_cleanup_report_v1.md`

Remaining top-level files:

- `docs/governance/docs_cleanup_report_v1.md`
- `docs/maintained_script_surface.tsv`
- `docs/mdec084_writer_contract_v1.md`
- `docs/run_directory_compliance_report.tsv`
- `docs/run_spec_template.md`
- `docs/SCRIPT_GOVERNANCE_POLICY.md`
- `docs/src_script_compliance_report.tsv`
- `docs/src_script_registry.tsv`
- `docs/tool_index.md`

`docs/` top-level is now low-noise on a best-effort basis. The remaining
top-level files are either high-reference registry/index surfaces or
governance-support docs that are still referenced directly from `project/`.

Files intentionally kept at top level in this pass:

- `docs/run_spec_template.md`
- `docs/mdec084_writer_contract_v1.md`
  - these remain in place because moving them would require coordinated edits
    to active `project/` governance docs, which this pass intentionally did not
    touch
- `docs/tool_index.md`
- `docs/maintained_script_surface.tsv`
- `docs/src_script_registry.tsv`
- `docs/src_script_compliance_report.tsv`
- `docs/run_directory_compliance_report.tsv`
  - these remain top-level because they function as high-reference navigation
    or registry surfaces

Remaining files that may still need human judgment:

- `docs/mdec084_writer_contract_v1.md`
- `docs/run_spec_template.md`
  - these still fit normalized buckets structurally, but they were left in
    place to avoid breaking `project/` references in this constrained pass
