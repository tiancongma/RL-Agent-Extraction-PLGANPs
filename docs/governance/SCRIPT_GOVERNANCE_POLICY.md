# Script Governance Policy

This document defines the repository-wide script legitimacy contract and run reproducibility contract.

It is authoritative together with:

- [AGENTS.md](../../AGENTS.md)
- [ACTIVE_PIPELINE_RUNBOOK.md](../../project/ACTIVE_PIPELINE_RUNBOOK.md)
- [PIPELINE_SCRIPT_MAP.md](../../project/PIPELINE_SCRIPT_MAP.md)

## Script Legitimacy Rule

Every `src/*.py` script must have an explicit recorded row in [src_script_registry.tsv](../src_script_registry.tsv).

The minimum required metadata is:

1. `script_path`
2. `status`
3. `architecture_layer`
4. `function_summary`
5. `primary_inputs`
6. `primary_outputs`
7. `upstream_dependencies`
8. `downstream_consumers`
9. `current_pipeline_role`
10. `evidence_source`
11. `confidence`
12. `disposition`

If a script cannot be classified clearly enough to fill these fields with defensible confidence, it is non-compliant. Non-compliant scripts must not continue to be treated as active engineering assets. They must be marked for one of the following dispositions:

- `keep_in_src`
- `keep_but_mark_branch_only`
- `move_to_archive`
- `delete_candidate_after_confirmation`

The current compliance state is recorded in [src_script_compliance_report.tsv](../src_script_compliance_report.tsv).

Wave-2 archive locations:

- archive-only historical code moved under `archive/code/`
- delete candidates pending confirmation moved under `archive/delete_candidates_pending_confirmation/`

## Run Reproducibility Rule

Every `data/results/run_*` directory must contain a reproducibility-grade run specification in its run root.

The run specification must include:

1. run purpose
2. run type
3. exact starting inputs and paths
4. exact script execution order
5. exact script paths used at each step
6. intermediate artifacts produced at each step
7. final outputs produced
8. benchmark-valid or diagnostic-only status
9. environment assumptions if needed
10. enough detail for manual rerun

The approved section template is [run_spec_template.md](../run_spec_template.md).

A run directory lacking this specification is non-compliant. A note file that only explains intent or outputs is not sufficient.

The current run-directory audit is recorded in [run_directory_compliance_report.tsv](../run_directory_compliance_report.tsv).

Wave-2 run-history segregation markers:

- [CURRENT_ENGINEERING_RUNS_INDEX.md](../../data/results/CURRENT_ENGINEERING_RUNS_INDEX.md)
- [HISTORICAL_NON_COMPLIANT_RUNS_INDEX.md](../../data/results/HISTORICAL_NON_COMPLIANT_RUNS_INDEX.md)
- physical historical run area: `data/results/historical_non_compliant_runs/`

## Enforcement Rule

- New scripts must be registered before they are treated as legitimate repo assets.
- New run directories must include a reproducibility-grade run spec at creation time.
- Undocumented scripts and undocumented runs are governance failures, not acceptable temporary shortcuts.
- Historical scripts may remain only if their archive/reference role is explicit and recorded.
