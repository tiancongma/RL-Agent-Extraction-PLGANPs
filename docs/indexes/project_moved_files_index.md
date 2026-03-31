# Project Moved Files Index

This index centralizes traceability for the non-authoritative files that were
removed from `project/` during the MDEC084 governance cleanup.

Under the current governance rule, `project/` is reserved for active
authoritative governance and contract documents only. Audit reports, diagnosis
notes, parking-lot material, open questions, temporary plans, and non-core
design/support documents were moved into `docs/` and are tracked here instead
of through distributed pointer stubs inside `project/`.

| Original path | New path | Category | Reason moved |
|---|---|---|---|
| `project/5_PARKING_LOT.md` | `docs/parking_lot/5_PARKING_LOT.md` | `parking_lot` | Parking-lot material is not an authoritative governance contract. |
| `project/memory_maintenance_audit_2026-03-27.md` | `docs/audits/memory_maintenance_audit_2026-03-27.md` | `audit_report` | Maintenance audit history belongs in `docs/`, not the governance layer. |
| `project/layer3_gt_cross_audit_report_v4.md` | `docs/audits/layer3_gt_cross_audit_report_v4.md` | `audit_report` | Audit findings are non-authoritative report material. |
| `project/layer3_cross_audit_runtime_diagnosis_2026-03-28.md` | `docs/audits/layer3_cross_audit_runtime_diagnosis_2026-03-28.md` | `diagnosis_note` | Runtime diagnosis notes are engineering history, not governance. |
| `project/blank_unloaded_drug_contamination_audit_2026-03-28.md` | `docs/audits/blank_unloaded_drug_contamination_audit_2026-03-28.md` | `audit_report` | Audit findings are non-authoritative report material. |
| `project/design/identity_freeze_and_attachment_rule_v1.md` | `docs/design/identity_freeze_and_attachment_rule_v1.md` | `design_contract` | Active design-support material is not part of the governance-only `project/` core. |
| `project/design/layer2_identity_scaffold_contract_v1.md` | `docs/design/layer2_identity_scaffold_contract_v1.md` | `design_contract` | Diagnostic design contract belongs under `docs/design/`. |
| `project/design/minimal_final_output_integration_plan.md` | `docs/working/minimal_final_output_integration_plan.md` | `temporary_plan` | Integration planning material is not authoritative governance. |
| `project/design/minimal_final_output_io_contract.tsv` | `docs/design/minimal_final_output_io_contract.tsv` | `design_contract` | Support contract table belongs under `docs/design/`, not `project/`. |
| `project/design/MINIMAL_FINAL_OUTPUT_LAYER_DESIGN.md` | `docs/design/MINIMAL_FINAL_OUTPUT_LAYER_DESIGN.md` | `design_contract` | Design reference material is non-core governance content. |
| `project/design/minimal_final_output_open_questions.md` | `docs/working/minimal_final_output_open_questions.md` | `open_question` | Open questions are working material, not governance. |
| `project/design/minimal_final_output_reference_assets.tsv` | `docs/design/minimal_final_output_reference_assets.tsv` | `design_contract` | Design-support reference assets belong in `docs/design/`. |
| `project/design/minimal_final_output_responsibility_split.tsv` | `docs/design/minimal_final_output_responsibility_split.tsv` | `design_contract` | Design-support responsibility mapping belongs in `docs/design/`. |

`project/` is now reserved for active authoritative governance only.
