# Repo Reduction Review

## 1. Scope

This is a dry-run repository reduction review for `project/`, `docs/`, and `data/`.
No deletions were performed. No files were renamed. No code behavior was changed.

## 2. Authoritative keep set

- `project/0_PROJECT_CHARTER.md` — primary project-scope contract.
- `project/1_REQUIREMENTS.md` — primary requirements contract.
- `project/2_ARCHITECTURE.md` — primary architecture and stage-boundary contract.
- `project/3_STATE_MACHINE.md` — primary lifecycle/state-transition contract.
- `project/4_DECISIONS_LOG.md` — primary durable decision history.
- `project/5_PARKING_LOT.md` — primary backlog for deferred work.
- `project/FILE_NAMING_AND_VERSIONING.md` — primary naming/run-id/layout policy.
- `project/PIPELINE_SCRIPT_MAP.md` — primary stage-to-script interpretation map.
- `project/ACTIVE_PIPELINE_FLOW.md` — primary active DEV-15 execution contract.
- `project/ACTIVE_PIPELINE_RUNBOOK.md` — primary active script registry and entrypoint guide.
- `project/AGENT_RUNBOOK.md` — primary stable agent execution contract.
- `project/7_DATASET_LAYOUT_CONVENTION.md` — primary dataset-scoped cleaned-asset layout policy, explicitly referenced by `project/FILE_NAMING_AND_VERSIONING.md`.
- `project/8_EVAL_SPLITS_REGISTRY.md` — only checked-in DEV split registry with exact exclusion keys.
- `docs/tool_index.md` — current reusable-tool inventory referenced by both runbooks.
- `docs/benchmarks/benchmark_goren_2025_engineering_spec.md` — primary benchmark/data-db engineering spec.
- `data/cleaned/labels/manual/dev15_formulation_skeleton/` — authoritative DEV-15 GT workbook family and validation outputs.
- `data/cleaned/labels/manual/formulation_instance_dev15_combined_eval_2026-03-10_reconciled.tsv` — current official reconciled DEV-15 combined count artifact named in `project/ACTIVE_PIPELINE_RUNBOOK.md` and `project/PIPELINE_SCRIPT_MAP.md`.
- `data/benchmark/goren_2025/overlap_goren18_v1/` — benchmark artifact root referenced by `docs/benchmarks/benchmark_goren_2025_engineering_spec.md`.
- `data/db/db_v1/` — current checked-in database snapshot referenced by `docs/benchmarks/benchmark_goren_2025_engineering_spec.md`.

## 3. Merge candidates

- `project/project_specification_UPDATED_20260201_v7.txt`; recommended merge target: `project/1_REQUIREMENTS.md`, `project/2_ARCHITECTURE.md`, `project/AGENT_RUNBOOK.md`; rationale: it claims to be the "single authoritative specification", but that role is now split across the modular governance docs and runbooks; original can likely be deleted after merge: yes.
- `project/project_specification_UPDATED_20260131_v6.txt`; recommended merge target: `project/1_REQUIREMENTS.md`, `project/2_ARCHITECTURE.md`, `project/AGENT_RUNBOOK.md`; rationale: same superseded monolithic-spec pattern as v7, with versioned historical content better preserved as durable requirements/decisions instead of separate copies; original can likely be deleted after merge: yes.
- `project/project_specification_UPDATED_20260130_v5.txt`; recommended merge target: `project/1_REQUIREMENTS.md`, `project/2_ARCHITECTURE.md`, `project/AGENT_RUNBOOK.md`; rationale: same superseded monolithic-spec pattern as v6/v7; original can likely be deleted after merge: yes.
- `docs/snapshots/snapshot_2026-03-06_llm_rule_audit_and_contract.md`; recommended merge target: `project/2_ARCHITECTURE.md` and `project/4_DECISIONS_LOG.md`; rationale: its substantive content is architecture/decision material and is already partially restated there; original can likely be deleted after merge: yes.
- `docs/snapshots/snapshot_2026-03-06_weak_labels_v7_schema_target.md`; recommended merge target: `project/2_ARCHITECTURE.md` and `project/4_DECISIONS_LOG.md`; rationale: it is an architecture-direction note, not an enduring standalone operational doc; original can likely be deleted after merge: yes.
- `docs/snapshots/snapshot_2026-03-08_formulation_grouping_and_record_assembly.md`; recommended merge target: `project/2_ARCHITECTURE.md` and `project/4_DECISIONS_LOG.md`; rationale: it summarizes the formulation-hypothesis architecture now captured more durably in governance docs; original can likely be deleted after merge: yes.
- `docs/methods/dev15_review_workbook_v1.md`; recommended merge target: `project/ACTIVE_PIPELINE_FLOW.md`; rationale: workbook purpose and I/O belong with the active flow, and this note still names the non-reconciled combined TSV; original can likely be deleted after merge: yes.
- `project/ACTIVE_PIPELINE_FLOW.tsv`; recommended merge target: `project/ACTIVE_PIPELINE_FLOW.md`; rationale: it duplicates the same DEV15 step table and commands in a second format while the markdown file is the named authoritative target; original can likely be deleted after merge: yes.
- `docs/audits/historical_script_triage_v1.tsv`; recommended merge target: `docs/tool_index.md` and `project/ACTIVE_PIPELINE_RUNBOOK.md`; rationale: any durable script-status conclusions should live in the maintained registry docs, not a dated triage export; original can likely be deleted after merge: yes.
- `docs/audits/historical_script_triage_priority.tsv`; recommended merge target: `docs/tool_index.md` and `project/ACTIVE_PIPELINE_RUNBOOK.md`; rationale: same as above, and it is a derivative priority view of the same triage pass; original can likely be deleted after merge: yes.
- `project/6_AGENT_RUNBOOK.md`; recommended merge target: `project/AGENT_RUNBOOK.md`; rationale: it is explicitly a compatibility pointer and not an authoritative runbook; inbound references should be updated to the canonical file before removal; original can likely be deleted after merge: uncertain.
- `project/ARCHIVED_BASELINES.md`; recommended merge target: `project/4_DECISIONS_LOG.md`; rationale: remote baseline provenance is durable historical knowledge, but it does not need a dedicated top-level doc if only one archived baseline is being tracked; original can likely be deleted after merge: uncertain.

## 4. Delete candidates

- `data/cleaned_backup_20260130/`; category: backup; evidence for deletion candidacy: top-level name is explicitly backup-like, it sits outside the canonical `data/cleaned/<dataset_id>/...` layout, and no active runbook or flow doc references it; risk level: medium; deletion recommendation: needs content check.
- `project/debug/dev3_grouping_debug_postprocess_doe_table.md`; category: debug artifact; evidence for deletion candidacy: stored under `project/debug/`, tied to a specific run path `run_20260309_1632_aa0bb8a_dev3_grouping_debug`, and not referenced by the active runbook/flow; risk level: low; deletion recommendation: safe after review.
- `project/debug/dev3_predicted_formulation_skeleton.tsv`; category: debug artifact; evidence for deletion candidacy: same dev3 debug family as above, only used by the adjacent debug note, not part of any documented active flow; risk level: low; deletion recommendation: safe after review.
- `project/debug/dev3_predicted_formulation_skeleton_comparison.tsv`; category: debug artifact; evidence for deletion candidacy: same dev3 debug family as above, comparison residue for a single debugging exercise, not part of any authoritative flow; risk level: low; deletion recommendation: safe after review.
- `project/debug/dev15_candidate_trace_UFXX9WXE.md`; category: debug artifact; evidence for deletion candidacy: one-DOI candidate trace note under `project/debug/`, not named in runbooks or active flow docs; risk level: low; deletion recommendation: safe after review.
- `docs/snapshots/snapshot_2026-02-20_goren_dev18_eval_and_opt.md`; category: snapshot; evidence for deletion candidacy: explicitly titled snapshot/handoff, records a dated working-tree state, and benchmark outputs it describes now live under authoritative benchmark docs and checked-in artifact roots; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/formulation_instance_pilot3_eval_2026-03-10/`; category: run residue; evidence for deletion candidacy: active flow keeps `formulation_instance_pilot3_eval_synthmethod_2026-03-10/` instead; this base pilot3 eval directory is not named in `project/ACTIVE_PIPELINE_RUNBOOK.md`; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/formulation_instance_pilot3_eval_blockpack_2026-03-10/`; category: run residue; evidence for deletion candidacy: dated variant directory tied to a blockpack experiment, while active flow only keeps the synthmethod pilot3 path; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/formulation_instance_pilot3_eval_tableheavy_rule_2026-03-10/`; category: run residue; evidence for deletion candidacy: dated variant directory tied to a table-heavy rule experiment, while active flow only keeps the synthmethod pilot3 path; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/l3h2rs2h_blockpack_audit_2026-03-10/`; category: debug artifact; evidence for deletion candidacy: DOI-specific audit pack with v1-only naming, referenced only by similarly dated method notes, not by active flow docs; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/l3h2rs2h_blockpack_audit_2026-03-10_v2/`; category: debug artifact; evidence for deletion candidacy: same single-DOI audit family, superseded by later sibling versions, not named by any active runbook; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/l3h2rs2h_blockpack_audit_2026-03-10_v3/`; category: debug artifact; evidence for deletion candidacy: same single-DOI audit family, still only tied to one debugging thread and method notes; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/l3h2rs2h_formulation_instance_eval_2026-03-10/`; category: debug artifact; evidence for deletion candidacy: one-DOI evaluation residue, not part of the documented DEV-15 default path; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/l3h2rs2h_formulation_instance_eval_2026-03-10_v2/`; category: debug artifact; evidence for deletion candidacy: same single-DOI evaluation family with explicit v2 suffix, not part of the documented DEV-15 default path; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/l3h2rs2h_regression_audit_2026-03-10/`; category: debug artifact; evidence for deletion candidacy: one-DOI regression audit directory with baseline/fixed comparison files, not referenced by the active flow or runbook; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/formulation_instance_dev15_combined_eval_2026-03-10_reconciled_diff.tsv`; category: duplicate; evidence for deletion candidacy: derivative diff file next to the authoritative reconciled combined TSV, with no active-doc reference found; risk level: low; deletion recommendation: safe after review.
- `data/cleaned/labels/manual/gt_audit__run_20260218_1334_4e58c55_sample20_evidence_v2_v2.xlsx`; category: duplicate; evidence for deletion candidacy: filename carries a repeated `_v2_v2` suffix and a sibling `_v2.xlsx` already exists; risk level: low; deletion recommendation: safe after review.
- `data/cleaned/labels/manual/dev15_extracted_formulation_view.xlsx`; category: debug artifact; evidence for deletion candidacy: reviewer/debug workbook style artifact not named in the current active flow or runbook; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/dev15_highrisk_audit_pack.xlsx`; category: debug artifact; evidence for deletion candidacy: high-risk audit pack workbook not named in the active flow or runbook; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/dev_v7pilot3_audit_pack.xlsx`; category: debug artifact; evidence for deletion candidacy: dated pilot audit workbook outside the documented current DEV-15 mainline artifacts; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/dev_v7pilot3_field_mapping_audit.xlsx`; category: debug artifact; evidence for deletion candidacy: field-mapping audit workbook for a pilot variant, not part of the current default flow; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/dev_v7pilot3_field_mapping_audit_r3_fixflat.xlsx`; category: debug artifact; evidence for deletion candidacy: pilot debug workbook tied to the retired `fixflat` variant naming; active path uses `fixparse`; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/dev_v7pilot3_scope_comparison_r2.xlsx`; category: debug artifact; evidence for deletion candidacy: scope-comparison workbook for an `r2` pilot variant not used by the active flow; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/dev_v7pilot3_scope_comparison_r3.xlsx`; category: debug artifact; evidence for deletion candidacy: scope-comparison workbook for an experimental pilot comparison, not part of the active flow; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/doi_10.1016.j.ejpb.2004.09.002_raw_field_mapping_audit.xlsx`; category: debug artifact; evidence for deletion candidacy: single-DOI raw-field audit workbook, not part of any authoritative flow doc; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/doi_10.1016_j.ejpb.2004.09.002_value_flow_audit.xlsx`; category: debug artifact; evidence for deletion candidacy: single-DOI value-flow audit workbook, not part of any authoritative flow doc; risk level: medium; deletion recommendation: needs content check.
- `data/cleaned/labels/manual/doi_10.1016_j.ejpb.2004.09.002_value_flow_audit_fixparse.xlsx`; category: debug artifact; evidence for deletion candidacy: same single-DOI value-flow audit family, tied to the pilot `fixparse` debugging thread rather than the stable documented flow outputs; risk level: medium; deletion recommendation: needs content check.

## 5. Uncertain items requiring human decision

- `docs/ee_coverage_rl/` — `docs/tool_index.md` explicitly labels this area as branch/run diagnostics, and `project/FEATURE_EE_COVERAGE_RL_SCOPE.md` says the content is branch-only, but it still contains benchmark status and methodology-adjacent notes. What is unclear: whether branch-only EE modeling history should remain in this repo or be collapsed into benchmark/decision docs before removal.
- `project/FEATURE_EE_COVERAGE_RL_SCOPE.md` — clearly branch-specific and not part of the current mainline governance stack, but it is still referenced by `docs/ee_coverage_rl/ee_coverage_rl__benchmark18_status.md`. What is unclear: whether this branch-scope contract still has active governance value.
- `data/cleaned/labels/manual/formulation_instance_remaining12_eval_2026-03-10/` — the reconciled sibling is authoritative, but `docs/methods/dev15_review_workbook_v1.md` and `project/ACTIVE_PIPELINE_FLOW.tsv` still reference the non-reconciled remaining-12 summary path. What is unclear: whether any workbook/script still reads this older directory.
- `data/cleaned/labels/manual/formulation_instance_dev15_combined_eval_2026-03-10.tsv` — active runbook says the reconciled TSV is official, but `docs/methods/dev15_review_workbook_v1.md` and the script-noted `NEEDS_CONFIRMATION` flow entry still point to the non-reconciled combined TSV. What is unclear: whether this file is still a live input for `build_dev15_review_workbook_v1.py`.
- `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1.xlsx` — the fixed workbook is authoritative in the runbooks, but the annotation-tool doc still points to the unfixed workbook. What is unclear: whether this original workbook is needed as provenance or has been fully superseded.
- `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_gt_v1.tsv` — the fixed GT TSV has an authoritative sibling, but the annotation-tool doc still names the unfixed export. What is unclear: whether the unfixed TSV must be retained as pre-correction provenance.

## 6. Immediate low-risk cleanup batch

- `project/debug/dev3_grouping_debug_postprocess_doe_table.md`
- `project/debug/dev3_predicted_formulation_skeleton.tsv`
- `project/debug/dev3_predicted_formulation_skeleton_comparison.tsv`
- `project/debug/dev15_candidate_trace_UFXX9WXE.md`
- `data/cleaned/labels/manual/formulation_instance_dev15_combined_eval_2026-03-10_reconciled_diff.tsv`
- `data/cleaned/labels/manual/gt_audit__run_20260218_1334_4e58c55_sample20_evidence_v2_v2.xlsx`

## 7. Summary counts

- authoritative keep items: 19
- merge candidates: 12
- delete candidates: 27
- uncertain items: 6
