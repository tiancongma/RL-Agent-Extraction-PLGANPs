# Run Naming Policy Audit

## Documents read

- `AGENTS.md`
- `README.md`
- `project/0_PROJECT_CHARTER.md`
- `project/1_REQUIREMENTS.md`
- `project/2_ARCHITECTURE.md`
- `project/4_DECISIONS_LOG.md`
- `project/ACTIVE_DATA_SOURCE_CONTRACT.md`
- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/FILE_NAMING_AND_VERSIONING.md`
- `project/PIPELINE_SCRIPT_MAP.md`
- `docs/SCRIPT_GOVERNANCE_POLICY.md`
- `docs/maintained_script_surface.tsv`
- `docs/methods/results_top_level_governance_v1.md`
- `docs/methods/results_lineage_normalization_pass.md`
- `docs/run_directory_compliance_report.tsv`
- `docs/run_spec_template.md`
- `docs/working/current_engineering_runs_backfill_plan.md`
- `data/results/ACTIVE_RUN.json`
- `data/results/CURRENT_ENGINEERING_RUNS_INDEX.md`
- `data/results/HISTORICAL_NON_COMPLIANT_RUNS_INDEX.md`
- `data/results/results_top_level_audit.tsv`
- `data/results/results_top_level_audit.md`
- `runs/RUN_TEMPLATE.md`
- `runs/BASELINE_RUN_CHECKLIST.md`
- `runs/latest.txt`
- `src/utils/paths.py`
- `src/utils/run_id.py`
- `src/utils/run_latest.py`
- `src/utils/run_preflight.py`
- `src/utils/active_data_source.py`
- `src/utils/audit_run_lineage_layout_v1.py`
- `src/utils/audit_results_top_level_semantics_v1.py`

Additional `project/` files were read where their names or contents made them relevant to project-governance classification:

- `project/5_PARKING_LOT.md`
- `project/blank_unloaded_drug_contamination_audit_2026-03-28.md`
- `project/layer3_cross_audit_runtime_diagnosis_2026-03-28.md`
- `project/layer3_gt_cross_audit_report_v4.md`
- `project/memory_maintenance_audit_2026-03-27.md`
- `project/design/MINIMAL_FINAL_OUTPUT_LAYER_DESIGN.md`
- `project/design/minimal_final_output_integration_plan.md`
- `project/design/minimal_final_output_open_questions.md`
- `project/design/identity_freeze_and_attachment_rule_v1.md`
- `project/feature_units/FEATURE_UNIT_GOVERNANCE.md`

## Facts from existing governance

1. `project/` is explicitly defined as the governance layer containing only authoritative project definitions. `AGENTS.md` also says agents must not create audit notes, experiment notes, temporary reports, or alternative specifications there, and sets a maximum of 9 governance files.
2. `README.md` and `project/1_REQUIREMENTS.md` place run-scoped outputs under `data/results/<run_id>/...`.
3. `project/FILE_NAMING_AND_VERSIONING.md` defines the current valid run ID regex as `^run_\d{8}_\d{4}_[0-9a-f]{7}_.+$`. Time-of-day and a descriptive suffix are both mandatory under that rule.
4. `project/2_ARCHITECTURE.md`, `project/ACTIVE_PIPELINE_FLOW.md`, and `project/ACTIVE_PIPELINE_RUNBOOK.md` all define lineage child placement under `data/results/<parent_run_id>/lineage/children/<ordered_role>/<child_run_id>/`.
5. `project/ACTIVE_PIPELINE_RUNBOOK.md` and `project/FILE_NAMING_AND_VERSIONING.md` also say artifact folders under a run root must not repeat the full `run_id` or timestamp/hash fragments and should be functional only. That conflicts with the lineage-child pattern above because `<child_run_id>` repeats a full run-like identifier below a run root.
6. `docs/methods/results_top_level_governance_v1.md` adds a broad top-level containment rule: one top-level parent per `(YYYYMMDD, short_commit)` family, with the earliest timestamp retained as parent.
7. `project/ACTIVE_DATA_SOURCE_CONTRACT.md` makes current `data/results` authority explicit: `--run-dir`, else `data/results/ACTIVE_RUN.json`, else hard error. `runs/latest.txt` is legacy only for current `data/results` workflows.
8. `docs/run_spec_template.md`, `AGENTS.md`, and `docs/SCRIPT_GOVERNANCE_POLICY.md` require every `data/results/run_*` directory to carry a reproducibility-grade `RUN_CONTEXT.md` or equivalent with run purpose, run type, starting inputs, script order, script paths, intermediate artifacts, final outputs, and benchmark-validity status.
9. `runs/BASELINE_RUN_CHECKLIST.md` and `src/utils/paths.py` still preserve the older `runs/` metadata path and `runs/latest.txt` convention.

## Observed current practice

### `project/`

- `project/` currently contains 14 top-level files and 25 files total, exceeding the stated maximum of 9 governance files.
- Besides core governance docs, `project/` currently holds audit, diagnosis, parking-lot, and open-question material:
  - `project/blank_unloaded_drug_contamination_audit_2026-03-28.md`
  - `project/layer3_cross_audit_runtime_diagnosis_2026-03-28.md`
  - `project/layer3_gt_cross_audit_report_v4.md`
  - `project/memory_maintenance_audit_2026-03-27.md`
  - `project/5_PARKING_LOT.md`
  - `project/design/minimal_final_output_open_questions.md`
  - `project/design/minimal_final_output_integration_plan.md`
- Those files describe audits, runtime diagnosis, deferred ideas, or unresolved questions rather than stable authoritative contracts.

### `runs/`

- Observed directory names are:
  - `run_20260130_0913_9436e6c_sample10`
  - `run_20260201_0927_bb13267_sample20`
- Both match the old regex and require time-of-day.
- The suffix is descriptive and effectively mandatory in practice.
- `runs/latest.txt` points to `run_20260309_1632_aa0bb8a_dev3_grouping_debug`, which no longer exists under `runs/`; this confirms `runs/` is now legacy metadata rather than a reliable active results index.

### `data/results/`

- Current top-level names are mixed across at least four patterns:
  - fully compliant old-style run IDs such as `run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1`
  - old-style names missing required parts, such as `run_20260310_dev15_remaining12_synthmethod`
  - names with non-matching date format, such as `run_2026-03-26_dev15_nvidia_full_pipeline_v1`
  - non-run review surfaces and loose artifacts at top level, such as `dev15_review`, `doe_coordinate_reconciliation_v1`, and multiple `.tsv/.jsonl` files
- Current top-level audit files under `data/results/` already lag reality:
  - `data/results/results_top_level_audit.tsv` classifies only 2 canonical parents and 4 frozen exceptions.
  - The current tree contains many more top-level run-like directories.
  - The gap is grounded in the audit script and registry logic requiring valid old-style run IDs; invalid or newer alternate names fall outside classification.
- A compliant sub-pattern already exists inside some lineage trees:
  - ordered cue folders such as `01_stage2_remaining10`, `02_stage5_remaining10_closure`, `04_stage5_variant_governance_replay`
  - these show the repo already uses short ordinal-plus-cue labels in some places
- But child executions still usually carry full old-style `run_*` names beneath those folders, for example:
  - `.../lineage/children/04_stage5_variant_governance_replay/run_20260313_2002_c4eccc8_dev15_stage5_variant_governance_replay_v1`
- Observed depth from top-level run bucket is high:
  - `run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1` reaches relative depth 10
  - `run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1` reaches relative depth 8
  - `run_20260312_1030_455ac37_targeted5_stage2_regression_v1` reaches relative depth 7

## Violations against current written rules

1. `project/` is not being used only for authoritative contracts.
   - Multiple files are clearly audits, diagnoses, parking-lot notes, or open questions.
   - This conflicts with `AGENTS.md` sections 2, 3, 6, and 7.
2. `project/` exceeds its own file-count cap.
   - Observed: 14 top-level files and 25 files total.
   - Written cap: maximum governance files allowed is 9.
3. Current run-root child layout conflicts with itself.
   - Architecture/runbook documents permit `lineage/children/<ordered_role>/<child_run_id>/`.
   - Naming/runbook documents forbid repeating `run_id` or timestamp/hash fragments below a run root.
   - The current tree repeatedly does both.
4. Current `data/results/` contains top-level run-like directories that do not match the written run ID regex.
   - Examples: `run_2026-03-26_dev15_nvidia_full_pipeline_v1`, `run_20260310_dev15_remaining12_synthmethod`, `run_20260325_1434_f17211_dev15_3paper_true_semantic_replacement_validation_no_llm_v1`
5. Current `data/results/` top level still contains loose non-run artifacts and review surfaces.
   - This conflicts with `docs/methods/results_top_level_governance_v1.md`, which says top level is an entry-point layer, not a dumping ground.
6. At least one current run root uses a non-standard run type string.
   - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/RUN_CONTEXT.md` uses `full_pipeline_benchmark_experiment`.
   - `README.md` and `docs/run_spec_template.md` only list accepted run types as `intermediate_diagnostic_run`, `component_regression_run`, and `full_pipeline_benchmark_run`.

## Violations against the new target policy

1. Top-level bucket names are not currently `YYYYMMDD_<short_hash>`.
   - Current top level uses `run_...` names, historical exceptions, and non-run directories.
2. Time-of-day is currently mandatory under the written regex and is widely present in practice.
   - Target policy removes time from bucket names.
3. Rich meaning is currently encoded heavily in folder names.
   - Examples include `dev15_identity_variable_preservation_exp_v1`, `dev15_current_merged_benchmark_v1`, `dev15_v2_l3_value_gt_annotation_v4_repaired2`.
4. Child folders are not normalized to ordinal-plus-one-cue only.
   - Some child folders already fit the target pattern.
   - Many others are full `run_*` names or multiword cue folders.
5. Total nesting depth frequently exceeds 4 from the top-level run bucket.
6. Several current utilities and audits assume old-style full run IDs, so the target scheme is not drop-in compatible.

## Risks of changing naming/layout

1. `src/utils/run_id.py` hardcodes the old regex and old builder format.
2. `src/utils/run_latest.py` parses old run IDs positionally with `parts[4]` and `parts[5:]`.
3. `src/utils/run_preflight.py` generates old-style IDs and writes them to `runs/latest.txt`.
4. `src/utils/active_data_source.py` rejects any `ACTIVE_RUN.json.active_run_id` that is not a valid old-style run ID, and requires `active_run_dir` basename equality.
5. `src/utils/audit_run_lineage_layout_v1.py` groups lineages by an old-style `run_<date>_<time>_<hash>` prefix.
6. `src/utils/audit_results_top_level_semantics_v1.py` classifies top-level results only when names satisfy the old run ID pattern or its derivative lineage-prefix pattern.
7. Support scripts and registries already reference literal old-style paths in docs and metadata.
8. Existing deeply nested lineages embed meaning in both the ordered folder and the full child run directory name. Flattening that without losing reproducibility requires a replacement explanation file and migration notes.

## Recommended migration policy

1. Freeze historical non-compliant runs in place or under `data/results/historical_non_compliant_runs/`.
2. For future runs only, separate:
   - bucket identity: `YYYYMMDD_<short_hash>`
   - child execution identity: ordinal-plus-cue folder such as `01_stage2`
   - rich meaning and reproducibility: mandatory local markdown file, plus `RUN_CONTEXT.md`
3. Keep `ACTIVE_RUN.json` as the sole machine authority for current `data/results` workflows, but expand it to point to the active bucket and exact child artifact paths during migration.
4. Update parser utilities before creating new-format active runs.
5. Normalize child folder grammar first, then top-level bucket grammar.
6. Preserve old run names inside explanation files or metadata rather than in future folder names.
7. Do not rewrite history unless a path is already marked as historical or non-compliant and is no longer an active authority surface.

## Clear answer: can the repo support the new naming scheme without breaking active authority?

Yes, but not as-is.

Grounded answer:

- The repo already has two ingredients the target scheme needs:
  - explicit active authority via `data/results/ACTIVE_RUN.json`
  - an existing partial child-folder convention using ordered short cues
- The current codebase also has several hard dependencies on the old full run ID grammar.
- Because of those dependencies, switching only the directory names would break current validators, latest-pointer helpers, lineage audits, and `ACTIVE_RUN.json` parsing.

So the repo can support the new naming scheme without breaking active authority only if it first makes targeted governance and utility changes, especially in:

- `src/utils/run_id.py`
- `src/utils/run_latest.py`
- `src/utils/run_preflight.py`
- `src/utils/active_data_source.py`
- `src/utils/audit_run_lineage_layout_v1.py`
- `src/utils/audit_results_top_level_semantics_v1.py`

Without those changes, the new naming scheme is not currently compatible with the repo's active authority enforcement.
