# Project Governance Cleanup Report v1

## Summary

This pass audited all files under `project/`, classified them against the
MDEC084 governance-only rule, and performed a minimal safe cleanup.

Before cleanup:

- scanned files under `project/`: `25`
- clearly non-governance files present in `project/`: `13`

After cleanup:

- authoritative live files retained in `project/`: `12`
- moved non-authoritative live files into `docs/`: `13`
- pointer stubs left behind in original `project/` paths: `13`

Important boundary:

- no files were deleted
- no scientific pipeline logic changed
- no run artifacts changed
- no content was rewritten beyond the required pointer stubs

## Classification Rules Used

- `governance_contract`
  - authoritative charter, requirements, architecture, naming, flow, runbook,
    active data-source contract, decisions log, pipeline script map, or
    explicitly governed feature-unit contracts
- `design_contract`
  - active or historical design/contract material that may still matter, but
    is not part of the governance-only core allowed in `project/`
- `audit_report`
  - audit findings, maintenance audits, or report-only review artifacts
- `diagnosis_note`
  - runtime or engineering diagnosis notes
- `parking_lot`
  - deferred idea backlog material
- `open_question`
  - unresolved design-question material
- `temporary_plan`
  - integration or working plans

## Moved Files

- `project/5_PARKING_LOT.md` -> `docs/parking_lot/5_PARKING_LOT.md`
- `project/memory_maintenance_audit_2026-03-27.md` -> `docs/audits/memory_maintenance_audit_2026-03-27.md`
- `project/layer3_gt_cross_audit_report_v4.md` -> `docs/audits/layer3_gt_cross_audit_report_v4.md`
- `project/layer3_cross_audit_runtime_diagnosis_2026-03-28.md` -> `docs/audits/layer3_cross_audit_runtime_diagnosis_2026-03-28.md`
- `project/blank_unloaded_drug_contamination_audit_2026-03-28.md` -> `docs/audits/blank_unloaded_drug_contamination_audit_2026-03-28.md`
- `project/design/identity_freeze_and_attachment_rule_v1.md` -> `docs/design/identity_freeze_and_attachment_rule_v1.md`
- `project/design/layer2_identity_scaffold_contract_v1.md` -> `docs/design/layer2_identity_scaffold_contract_v1.md`
- `project/design/minimal_final_output_integration_plan.md` -> `docs/working/minimal_final_output_integration_plan.md`
- `project/design/minimal_final_output_io_contract.tsv` -> `docs/design/minimal_final_output_io_contract.tsv`
- `project/design/MINIMAL_FINAL_OUTPUT_LAYER_DESIGN.md` -> `docs/design/MINIMAL_FINAL_OUTPUT_LAYER_DESIGN.md`
- `project/design/minimal_final_output_open_questions.md` -> `docs/working/minimal_final_output_open_questions.md`
- `project/design/minimal_final_output_reference_assets.tsv` -> `docs/design/minimal_final_output_reference_assets.tsv`
- `project/design/minimal_final_output_responsibility_split.tsv` -> `docs/design/minimal_final_output_responsibility_split.tsv`

Each original path now contains a small pointer stub so link targets are not
broken silently in this pass.

## Files Still Questionable

- `project/0_PROJECT_CHARTER.md`
  - retained because it is an intended governance slot, but the file is
    currently empty and should be reviewed separately
- `project/feature_units/FEATURE_UNIT_GOVERNANCE.md`
- `project/feature_units/feature_intervention_matrix.tsv`
- `project/feature_units/feature_unit_registry.json`
  - retained because they read as explicitly governed feature-unit contract
    surfaces, which the user explicitly allowed
  - together they push the live authoritative `project/` file count above the
    nominal `9`-file governance expectation recorded in `AGENTS.md`

## Confirmation On Authoritative Contracts

No core authoritative governance contracts were moved incorrectly.

Retained in `project/` as live authoritative surfaces:

- `0_PROJECT_CHARTER.md`
- `1_REQUIREMENTS.md`
- `2_ARCHITECTURE.md`
- `4_DECISIONS_LOG.md`
- `ACTIVE_DATA_SOURCE_CONTRACT.md`
- `ACTIVE_PIPELINE_FLOW.md`
- `ACTIVE_PIPELINE_RUNBOOK.md`
- `FILE_NAMING_AND_VERSIONING.md`
- `PIPELINE_SCRIPT_MAP.md`
- `feature_units/FEATURE_UNIT_GOVERNANCE.md`
- `feature_units/feature_intervention_matrix.tsv`
- `feature_units/feature_unit_registry.json`

## Best-Effort Result

Best-effort outcome:

- yes, the live substantive content left in `project/` is governance-level or
  explicitly governed feature-unit contract material
- no, `project/` does not yet contain only governance-level files in the
  strict physical sense, because this pass intentionally leaves non-authoritative
  pointer stubs behind at moved paths to preserve traceability and avoid
  breaking links silently

## Physical Cleanup Pass

This follow-up pass removed the distributed pointer stubs from `project/` after
confirming that each one matched the prior stub template and that its moved
destination file still existed.

- pointer stubs found: `13`
- pointer stubs removed: `13`
- centralized traceability index added:
  - `docs/indexes/project_moved_files_index.md`
- remaining files in `project/`: `12`

Remaining `project/` files after stub removal:

- `0_PROJECT_CHARTER.md`
- `1_REQUIREMENTS.md`
- `2_ARCHITECTURE.md`
- `4_DECISIONS_LOG.md`
- `ACTIVE_DATA_SOURCE_CONTRACT.md`
- `ACTIVE_PIPELINE_FLOW.md`
- `ACTIVE_PIPELINE_RUNBOOK.md`
- `FILE_NAMING_AND_VERSIONING.md`
- `PIPELINE_SCRIPT_MAP.md`
- `feature_units/FEATURE_UNIT_GOVERNANCE.md`
- `feature_units/feature_intervention_matrix.tsv`
- `feature_units/feature_unit_registry.json`

Physical governance-only result:

- yes, `project/` is now physically governance-only in appearance because the
  distributed non-authoritative pointer files were removed
- traceability is preserved through the centralized moved-files index rather
  than through stub placeholders inside `project/`

Remaining borderline files for human review:

- `project/0_PROJECT_CHARTER.md`
  - still empty, but retained as an intended governance slot rather than moved
- `project/feature_units/FEATURE_UNIT_GOVERNANCE.md`
- `project/feature_units/feature_intervention_matrix.tsv`
- `project/feature_units/feature_unit_registry.json`
  - these still read as authoritative governed contract material, but they
    keep the live `project/` file count above the nominal `9`-file governance
    expectation stated in `AGENTS.md`
