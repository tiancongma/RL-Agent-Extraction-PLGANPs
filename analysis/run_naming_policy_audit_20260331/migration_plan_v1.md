# Migration Plan v1

## Immediate no-code governance changes

1. Resolve the internal contradiction between:
   - `lineage/children/<ordered_role>/<child_run_id>/`
   - and the rule forbidding repeated `run_id` tokens below a run root
2. Adopt one future-only naming rule for `data/results/` buckets and child folders.
3. Keep `ACTIVE_RUN.json` as the explicit authority source.
4. Freeze historical non-compliant runs and label them clearly instead of retrofitting them immediately.
5. Move non-authoritative material out of `project/` into governed `docs/` locations.

Docs likely needing edits first:

- `project/FILE_NAMING_AND_VERSIONING.md`
- `project/2_ARCHITECTURE.md`
- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `docs/methods/results_top_level_governance_v1.md`
- `docs/methods/results_lineage_normalization_pass.md`
- `README.md`

## Low-risk script changes

1. Refactor `src/utils/run_id.py`
   - add a v2 bucket validator/builder
   - keep a historical old-style parser for backward reads
2. Refactor `src/utils/run_latest.py`
   - stop deriving semantic metadata from old token positions
   - store explicit metadata instead
3. Refactor `src/utils/run_preflight.py`
   - create or reuse a v2 bucket
   - allocate the next ordinal child folder
4. Refactor `src/utils/audit_run_lineage_layout_v1.py`
   - group by bucket ID rather than old-style run prefix
5. Refactor `src/utils/audit_results_top_level_semantics_v1.py`
   - classify v2 buckets, archives, and review surfaces explicitly

## High-risk migrations

1. Changing `src/utils/active_data_source.py`
   - this touches current authority enforcement
   - update only after the new `ACTIVE_RUN.json` schema is settled
2. Rewriting active lineage directories
   - any move under a currently referenced active run risks breaking literal path references in docs, registries, and run contexts
3. Flattening deeply nested historical lineages
   - current repo history uses nested child-run folders as meaning carriers
   - flattening them safely needs migration metadata and updated references
4. Updating any script that assumes old-style names in literals or docs
   - for example diagnostics and branch-only utilities that reference literal old paths

## Historical runs that should remain frozen

- Entries already indexed by `data/results/HISTORICAL_NON_COMPLIANT_RUNS_INDEX.md`
- Top-level non-compliant legacy names such as:
  - `run_20260310_dev15_remaining12_synthmethod`
  - `run_20260310_dev15_remaining12_synthmethod_merged`
  - `run_20260310_v7pilot3r3fixparse_synthmethod`
  - `run_2026-03-26_dev15_nvidia_full_pipeline_v1`
  - related `run_2026-03-26_*` NVIDIA diagnostic directories

Rationale:

- They already violate current rules or belong to older naming eras.
- Freezing them avoids rewriting historical evidence just to fit v2.

## Exact scripts likely needing edits

Highest priority:

- `src/utils/run_id.py`
- `src/utils/run_latest.py`
- `src/utils/run_preflight.py`
- `src/utils/active_data_source.py`
- `src/utils/audit_run_lineage_layout_v1.py`
- `src/utils/audit_results_top_level_semantics_v1.py`

Second wave:

- `src/utils/validate_active_data_source_contract_v1.py`
- `src/analysis/dev15_relation_diagnostics_v1.py`
- any future script that still parses `run_YYYYMMDD_HHMM_<hash>_...` directly

Branch/supporting surfaces to inspect for literal old-path dependencies:

- `src/utils/run_dev15_nvidia_full_pipeline_v1.py`
- `src/utils/evaluate_nvidia_full_pipeline_against_dev15_gt.py`
- `src/stage5_benchmark/validate_stage5_descendant_filter_regression_v1.py`

## Minimal-change migration order

1. Update governance docs and agree on v2 schema.
2. Make readers accept both old and new naming.
3. Change `ACTIVE_RUN.json` validation to accept the new authority form.
4. Start writing only v2 names for future runs.
5. Leave historical runs frozen unless a later cleanup task explicitly migrates them.
