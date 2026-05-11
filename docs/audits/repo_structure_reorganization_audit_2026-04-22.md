# Repo Structure Reorganization Audit 2026-04-22

## Scope

This audit covers three user-raised disorder zones:

1. repository root top-level file clutter
2. `analysis/` root-level sprawl and unclear placement policy
3. `docs/` top-level loose files outside normalized buckets

This audit does not move files yet. It defines a concrete target structure, a move policy, and guardrails to prevent recurrence.

## Governance constraints used

The proposal is constrained by the current repository contracts:

- `project/AGENTS.md` governance rules: keep `project/` authoritative and minimal, keep layers separated, avoid undocumented sprawl
- `project/1_REQUIREMENTS.md`: stable filesystem contracts preferred over path churn
- `project/2_ARCHITECTURE.md`: `data/results/<run_id>/...` is run-scoped output; docs and analysis should not impersonate pipeline outputs
- `project/ACTIVE_PIPELINE_RUNBOOK.md`: new result lineages must keep execution meaning inside `RUN_CONTEXT.md`; functional artifact folders under runs should include `analysis/`, `outputs/`, `audit/`
- `project/FILE_NAMING_AND_VERSIONING.md`: names must not encode ad hoc trial-and-error semantics when governed placement can encode meaning more cleanly

## Current-state findings

### A. Repository root is carrying non-root material

Observed top-level files before cleanup that did not belong at root:

- `anti_patterns.md`
- `changed_files.tsv`
- `enforcement_plan.md`
- `execution_output.txt`
- `failure_mechanism_analysis.md`
- `old_vs_new_execution_path.md`
- `print_tree.py`
- `project_specification.txt`
- `project_structure.txt`
- `run_stage1_py313.bat`
- `safeguard_design.md`

These are mixed across several different roles:

- one-off analysis/report artifacts
- local utility or helper scripts
- legacy project-definition text now superseded by governed `project/*.md`
- local execution output scratch files
- local platform-specific helper launcher

Problem:
- root should expose stable entry surfaces and major layer directories only
- these files increase ambiguity about what is authoritative, reusable, local-only, or historical

### B. `analysis/` is being used as a catch-all instead of a typed surface

Observed state:

- `analysis/` has 146 root-level files and only 4 subdirectories
- root-level file mix includes audits, repair reports, baseline diffs, prompt audits, mem validation, docs cleanup inventories, writer rollout checks, etc.
- many files naturally cluster by purpose but are flattened together

The root-level `analysis/` directory currently mixes at least these categories:

1. repo governance / cleanup audits
2. Stage1 validation experiments
3. Stage2 contract and prompt audits
4. repair reports for specific papers
5. baseline comparison and regression localization
6. memory-system validation
7. writer / results layout migration audits
8. synthetic validation fixtures

Problem:
- difficult to browse
- names alone must carry too much meaning
- overlap with `docs/audits/` and `data/results/*/analysis/` is unclear
- some analysis artifacts are repo-level durable references, while others are run-specific and should live under the corresponding lineage

### C. `docs/` is improved but still not fully normalized

Observed `docs/` top-level loose files before cleanup:

- `dev15_llm_capability_audit_summary.md`
- `dev15_llm_capability_audit_v1.tsv`
- `docs_cleanup_report_v1.md`
- `doe_scope_condition_matrix_v1.json`
- `doe_scope_construction_audit_v1.md`
- `inmutv7l_explanation_audit_v1.md`
- `maintained_script_surface.tsv`
- `mdec084_writer_contract_v1.md`
- `run_directory_compliance_report.tsv`
- `run_spec_template.md`
- `sequential_optimization_fu_validation.md`
- `src_script_compliance_report.tsv`
- `src_script_registry.tsv`
- `sync_strategy.md`
- `tool_index.md`

Existing cleanup records already identified part of this problem:

- `analysis/repo_governance/docs_cleanup_inventory.tsv`
- `docs/governance/docs_cleanup_report_v1.md`

Problem:
- `docs/` top level still mixes navigation, registries, governance-support references, audits, methods, and domain-specific notes
- top-level should be reserved for high-reference navigation and registries only
- several files already have obvious bucket destinations

### D. Cross-cutting hygiene issue: path-portable links are inconsistent

`docs/indexes/DOCS_INDEX.md` and multiple other docs still contain hardcoded Windows absolute links like `/c:/Users/...`.

Problem:
- weak repository portability
- documentation navigation appears organized but is not actually repo-relative
- any cleanup pass should avoid creating more machine-local path bindings

## Structural decision model

The repository should use four placement tests before any new non-code artifact is created:

1. Is it authoritative project contract?
   - yes -> `project/`
2. Is it run-scoped execution output or run-scoped audit evidence?
   - yes -> under the corresponding `data/results/<lineage>/<child>/...`
3. Is it durable cross-run supporting documentation or reusable reference?
   - yes -> `docs/<typed_bucket>/`
4. Is it temporary local scratch, machine-local output, or operator helper?
   - yes -> `scripts/local/`, `.local/`, or ignore it from git

If a file fails all four tests, it should not exist.

## Target structure

### 1. Target repository root policy

Root should contain only:

- stable repo entry files
  - `AGENTS.md`
  - `README.md`
  - `LICENSE`
  - `requirements.txt`
  - `.gitignore`
- major layer directories
  - `project/`
  - `src/`
  - `data/`
  - `docs/`
  - `archive/`
  - `configs/`
  - `scripts/`
  - existing specialized code/service dirs such as `mcp_server/`
- hidden local config files explicitly intended to be root-local
  - `.env`
  - `.local_stage1_paths.json`
  - similar local config surfaces

Everything else should be relocated.

Recommended root additions:

- `scripts/local/` for operator-only helper launchers and inspection scripts
- optional `.local/` for untracked local scratch outputs if the repo wants a visible local sandbox

### 2. Target `analysis/` policy

Decision:

Keep a repo-level `analysis/` directory, but stop using it as a flat dump.

Rationale:
- some artifacts are genuinely cross-run engineering analysis and do not belong inside any single `data/results/*` lineage
- moving all root `analysis/` content into `docs/` would mix narrative documents with machine-oriented comparison tables and validation artifacts
- moving all analysis under `data/results/` would be wrong for repo-wide audits, migration checks, and cleanup inventories

Target meaning of root `analysis/`:
- repo-level, cross-run, engineering analysis artifacts only
- never run-scoped outputs
- never top-level flat dumps

Required sub-buckets under `analysis/`:

- `analysis/repo_governance/`
  - cleanup inventories
  - root/docs/project structure audits
  - results writer audits
  - pointer or authority audits
- `analysis/results_layout/`
  - run naming audits
  - writer rollout checks
  - lineage normalization studies
- `analysis/stage1_validation/`
  - py313 / py314 validation bundles and local stage1 validation artifacts
- `analysis/stage2_audits/`
  - prompt audits
  - evidence quality audits
  - table preservation audits
  - contract gate checks
- `analysis/baseline_regressions/`
  - baseline before/after screens
  - root-cause localizations
  - historical capability inventories
- `analysis/paper_repairs/<paper_key>/`
  - paper-specific repair reports such as `5zxyabsu_*`, `wfdtq4vx_*`, `inmutv7l_*`
- `analysis/memory_system/`
  - mem validation and failure traces
- `analysis/synthetic_validation/`
  - synthetic lineage layout checks like `mdec084_synthetic_v2_validation/`

Placement rule:
- if an artifact can be described as "for one run lineage", place it under `data/results/<that lineage>/analysis/` instead
- if an artifact compares multiple runs, multiple papers, or repo contracts, keep it under root `analysis/` in a typed sub-bucket

### 3. Target `docs/` policy

Target meaning of top-level `docs/`:
- high-reference navigation and registry surfaces only

Allowed long-term `docs/` top-level files:

- `tool_index.md`
- `maintained_script_surface.tsv`
- `src_script_registry.tsv`
- `src_script_compliance_report.tsv`
- `run_directory_compliance_report.tsv`
- optionally `run_spec_template.md` if project governance continues to reference it directly

Everything else should live under typed subdirectories.

Recommended `docs/` bucket model:

- `docs/audits/`
- `docs/methods/`
- `docs/design/`
- `docs/governance/`
- `docs/indexes/`
- `docs/working/`
- `docs/benchmarks/`
- `docs/snapshots/`
- `docs/repair_index/`
- `docs/feature_governance/`
- `docs/archive_project/`
- specialized topical buckets only when the topic is stable and recurring

## Recommended move plan

### A. Root -> target destinations

- `docs/governance/anti_patterns.md` -> `docs/governance/docs/governance/anti_patterns.md`
- `docs/working/enforcement_plan.md` -> `docs/working/docs/working/enforcement_plan.md`
- `docs/audits/failure_mechanism_analysis.md` -> `docs/audits/docs/audits/failure_mechanism_analysis.md`
- `docs/governance/old_vs_new_execution_path.md` -> `docs/governance/docs/governance/old_vs_new_execution_path.md`
- `docs/design/safeguard_design.md` -> `docs/design/docs/design/safeguard_design.md`
- `docs/indexes/changed_files.tsv` -> `docs/indexes/docs/indexes/changed_files.tsv` if it is a durable move/change ledger, otherwise archive or remove after use
- `scripts/local/print_tree.py` -> `scripts/local/scripts/local/print_tree.py`
- `scripts/local/run_stage1_py313.bat` -> `scripts/local/scripts/local/run_stage1_py313.bat`
- `docs/archive_project/project_specification_legacy.txt` -> `docs/archive_project/project_specification_legacy.txt`
- `analysis/repo_governance/project_structure_snapshot.txt` -> do not keep as a committed root file; either move to `analysis/repo_governance/project_structure_snapshot.txt` if actively referenced, or delete/regenerate on demand
- `execution_output.txt` -> remove from git and ignore as local scratch

### B. `analysis/` root -> typed buckets

Examples from the current root:

Repo governance / cleanup:
- `docs_cleanup_inventory.tsv` -> `analysis/repo_governance/docs_cleanup_inventory.tsv`
- `docs_post_cleanup_check.tsv` -> `analysis/repo_governance/docs_post_cleanup_check.tsv`
- `docs_top_level_finish_check.tsv` -> `analysis/repo_governance/docs_top_level_finish_check.tsv`
- `project_governance_cleanup_inventory.tsv` -> `analysis/repo_governance/project_governance_cleanup_inventory.tsv`
- `project_governance_post_cleanup_check.tsv` -> `analysis/repo_governance/project_governance_post_cleanup_check.tsv`
- `repo_sync_audit_2026-04-21_baseline.*` -> `analysis/repo_governance/`

Results layout / writer policy:
- `mdec084_default_writer_rollout_*` -> `analysis/results_layout/`
- `results_writer_audit_2026-04-15.md` -> `analysis/results_layout/`
- `results_write_guard_implementation_2026-04-15.md` -> `analysis/results_layout/`
- existing `run_naming_policy_audit_20260331/` stays under `analysis/results_layout/run_naming_policy_audit_20260331/`

Stage1 validation:
- `stage1_py313_validation/` -> `analysis/stage1_validation/py313_validation/`
- `stage1_py314_validation/` -> `analysis/stage1_validation/py314_validation/`
- `dev15_clean_text_*` -> `analysis/stage1_validation/`

Stage2 audits:
- `dev15_s2_4a_prompt_audit.*` -> `analysis/stage2_audits/`
- `s2_2_*`, `s2_3_*`, `s2_4a_*`, `s2_7_*` -> `analysis/stage2_audits/`
- `gemini_live_call_path_audit.*` -> `analysis/stage2_audits/`

Baseline regressions:
- `baseline_*` -> `analysis/baseline_regressions/`
- `post_baseline_*` -> `analysis/baseline_regressions/`
- `current_baseline_vs_gt_full_count_screen.*` -> `analysis/baseline_regressions/`
- `historical_capability_regression_inventory.*` -> `analysis/baseline_regressions/`

Paper repairs:
- `5zxyabsu_*` -> `analysis/paper_repairs/5zxyabsu/`
- `wfdtq4vx_*` -> `analysis/paper_repairs/wfdtq4vx/`
- `inmutv7l_*` -> `analysis/paper_repairs/inmutv7l/`
- `qlyk_*` -> `analysis/paper_repairs/qlyk/`
- `ufxx9wxe_*` -> `analysis/paper_repairs/ufxx9wxe/`

Memory system:
- `mem_*` -> `analysis/memory_system/`
- `auto_update_contract_for_repair_patterns_v1.md` -> likely `analysis/memory_system/` or `docs/methods/` depending on whether it is normative or analytical

Synthetic validation:
- `mdec084_synthetic_v2_validation/` -> `analysis/synthetic_validation/mdec084_synthetic_v2_validation/`

### C. `docs/` top-level -> target destinations

Recommended moves:

- `dev15_llm_capability_audit_summary.md` -> `docs/audits/dev15_llm_capability_audit_summary.md`
- `dev15_llm_capability_audit_v1.tsv` -> `docs/audits/dev15_llm_capability_audit_v1.tsv`
- `docs_cleanup_report_v1.md` -> `docs/governance/docs_cleanup_report_v1.md` or `docs/audits/docs_cleanup_report_v1.md`
- `doe_scope_condition_matrix_v1.json` -> `docs/design/doe_scope_condition_matrix_v1.json`
- `doe_scope_construction_audit_v1.md` -> `docs/audits/doe_scope_construction_audit_v1.md`
- `inmutv7l_explanation_audit_v1.md` -> `docs/audits/inmutv7l_explanation_audit_v1.md`
- `sequential_optimization_fu_validation.md` -> `docs/audits/sequential_optimization_fu_validation.md` if retrospective validation, or `docs/methods/` if reusable procedure
- `sync_strategy.md` -> `docs/governance/sync_strategy.md` if it is repo policy, or `docs/working/sync_strategy.md` if it is provisional

Likely keep at top level for now:

- `tool_index.md`
- `maintained_script_surface.tsv`
- `src_script_registry.tsv`
- `src_script_compliance_report.tsv`
- `run_directory_compliance_report.tsv`
- `run_spec_template.md` only if governance docs still reference this exact path
- `mdec084_writer_contract_v1.md` only if governance docs still reference this exact path

## Prevention rules

### Rule 1. No new non-hidden root files without category justification

Before creating a new top-level file, require one of:
- repo entry surface
- root-local config surface
- user-explicit request for root placement

Otherwise reject or relocate.

### Rule 2. Run-scoped analysis must live with the run

If an artifact was produced from a single run lineage or a single bounded replay, it must be written under that lineage’s `analysis/` or `audit/` folder, not under root `analysis/`.

### Rule 3. Root `analysis/` must be typed, never flat

Any new repo-level analysis file must be written into a typed subdirectory. Writing directly to `analysis/<file>` should be considered non-compliant except for a short-lived migration period.

### Rule 4. `docs/` top level is a whitelist, not a dumping ground

New docs go into sub-buckets by default. Only navigation and registry files may remain directly under `docs/`.

### Rule 5. Distinguish normative vs analytical docs

- normative reusable support docs -> `docs/governance/`, `docs/design/`, `docs/methods/`
- retrospective investigation or validation -> `docs/audits/`
- provisional plan / incomplete thinking -> `docs/working/`

### Rule 6. Local-only artifacts must not be committed casually

Examples:
- machine-local execution output
- platform-specific launcher scratch files
- generated tree snapshots

These should be either:
- moved to `scripts/local/`
- written under an ignored `.local/`
- or not committed at all

### Rule 7. Avoid machine-absolute links in docs

Use repo-relative paths only. Absolute `/c:/Users/...` links should be corrected when touched.

## Minimal enforcement mechanism

Recommended low-cost enforcement:

1. Add a small repository layout check script under `scripts/` or `src/utils/` that fails if:
   - root contains unauthorized non-hidden files
   - `analysis/` root receives new files outside an allowlist
   - `docs/` top level receives non-whitelisted files
2. Add the policy summary to `README.md` and/or `docs/governance/`
3. Add `.gitignore` rules for obvious local scratch such as:
   - `execution_output.txt`
   - `.local/`
   - similar transient outputs
4. When moving files, update `docs/indexes/docs_moves_index.tsv` so cross-reference repair is explicit

## Execution order for cleanup

Recommended order if this plan is executed:

1. clean repository root first
2. create typed `analysis/` sub-buckets and move current root files into them
3. finish `docs/` top-level normalization
4. repair broken cross-references and absolute links
5. add lightweight structural guardrail checks

## Decision summary

- Root: should be aggressively minimized
- `analysis/`: should remain, but only as typed repo-level cross-run analysis
- Run-specific analysis: should move into `data/results/<lineage>/analysis/`
- `docs/`: should reserve top-level for indexes, registries, and a very small set of governance-support entry surfaces
- Prevent recurrence by whitelisting top-level surfaces and enforcing placement rules
