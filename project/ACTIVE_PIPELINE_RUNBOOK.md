# Active Pipeline Runbook

This runbook explains how to execute the canonical pipeline manually, stage by
stage.

It does not define a hidden orchestration layer. The authoritative execution
order lives in `project/ACTIVE_PIPELINE_FLOW.md`, and this file explains how to
apply that flow in practice with correct run discipline, provenance, and
incremental reuse.

This runbook distinguishes:

- the production path
- the optional diagnostic / review path
- the evaluation reference path
- the final comparison node

## Read Order

1. `project/ACTIVE_PIPELINE_FLOW.md`
2. `project/PIPELINE_SCRIPT_MAP.md`
3. `project/4_DECISIONS_LOG.md`
4. `docs/tool_index.md`
5. `project/ACTIVE_DATA_SOURCE_CONTRACT.md` when resolving current
   `data/results` workflow sources

## Recent Changes (2026-03-19)

- Canonical polymer MW field:
  - `polymer_mw_kDa` is now the canonical field name.
  - `plga_mw_kDa` is retained only as a legacy read alias for compatibility with older artifacts.
  - This is a naming correction only; the field meaning did not change.
- Stage 2 contract:
  - Stage 2 now refers to the paper-driven semantic-object emitter as the
    authoritative semantic boundary.
  - The deterministic compatibility adapter is the required bridge into the
    legacy wide-row surface used by unchanged Stage3, Stage4 diagnostic, and
    Stage5 runtime consumers.
- Non-change reminder:
  - Stage 5 remains materialization-only.
  - The relation-first Stage 3 -> Stage 5 contract is unchanged.
- Legacy note:
  - The deprecated wide-row fallback extractor may still carry older evidence-
    packing behavior, but that is no longer the active Stage2 mainline.

## Operating Principle

A complete pipeline run means the declared scope has passed through every
required production stage from upstream corpus inputs to:

- `final_formulation_table_v1.tsv`

Benchmark-valid reporting then additionally requires the comparison node, which
reads:

- the final formulation table
- the fixed manual GT workbook
- the declared scope manifest

Complete pipeline does not mean forced full recomputation.

## Complete Pipeline vs Full Recomputation

These are not the same thing.

- Complete pipeline means every required canonical stage is covered by a valid,
  traceable artifact chain.
- Full recomputation means every upstream stage is re-run from scratch.

Allowed:

- reuse unchanged Stage 0 raw records
- reuse unchanged Stage 1 cleaned assets
- reuse unchanged Stage 2 candidate TSVs

Not allowed:

- skip a required stage entirely
- substitute undocumented files for canonical stage completion artifacts
- report a partial-layer artifact as if it were the final system result

Run layout rule:

- a run root is the directory named by `run_id`
- artifact folders under that run root must be functional only, such as:
  - `analysis/`
  - `outputs/`
  - `audit/`
  - stage-local functional folders like `formulation_relation_v1/`
- artifact folders under a run root must not repeat:
  - the full `run_id`
  - timestamp/hash fragments
- if a unit needs independent rerun or lineage identity, create a separate run
  root with its own `run_id` instead of nesting another run-like directory

Active data-source rule:

- Do not infer the current source run by sort order, modification time, parent
  fallback, or glob matching.
- Resolve source artifacts only from:
  - explicit CLI source paths such as `--run-dir`
  - or the repository authority pointer in `data/results/ACTIVE_RUN.json`
- If neither is available, fail loudly.

## Maintained Script Entrypoints

Default execution-facing benchmark, alignment, comparison, workbook-generation,
and audit workflows must use only the maintained entrypoints listed in:

- `docs/maintained_script_surface.tsv`

Selection rule:

- do not choose scripts by filename similarity, recency, or convenience
- do not auto-select wrapper-only, legacy, deprecated, or diagnostic scripts
- if the registry marks `must_use_active_data_source_contract=yes`, the script
  must resolve sources by:
  - explicit `--run-dir`
  - otherwise `data/results/ACTIVE_RUN.json`
  - otherwise hard error

## Supporting Memory Layer

The governed long-term memory subsystem lives under:

- `data/mem/v1/`

Rules:

- this memory surface is supporting only; it is not Stage 0, 1, 2, 3, 4, or 5
- before complex debugging or repeated failure-localization work, identify the task class and query memory first
- default bootstrap helper:
  - `src/utils/mem_bootstrap_v1.py`
- direct query tool:
  - `src/utils/query_mem_v1.py`
- standard order for complex tasks:
  1. query memory
  2. inspect top memory-linked files
  3. read local source files and active artifacts
  4. act
- rebuild memory with `src/utils/build_mem_v1.py` when governed source markdown or `RUN_CONTEXT.md` artifacts change materially
- use `src/utils/update_mem_v1.py` only for small manual additions or corrections that should append to the existing memory tables
- validate the memory surface with `src/utils/check_mem_v1.py` after rebuilds or manual updates

Recommended task-to-query pattern:

- debugging:
  - `collapse`
  - paper key if known
  - stage name if known
- regression investigation:
  - `regression`
  - affected benchmark scope or run theme
  - the specific failure phrase
- run comparison:
  - `run lineage`
  - both run themes or run IDs
- pipeline modification:
  - intended behavior phrase such as `family variant` or `table-first`
  - affected stage name
- GT mismatch analysis:
  - paper key or mismatch phrase such as `identity mismatch`
- lineage tracing:
  - `run lineage`
  - parent or child run theme

Maintained-surface update rule:

- when a maintained execution entrypoint changes, update both this runbook and
  `docs/maintained_script_surface.tsv` in the same change
- if an older script remains in `src/` for historical, wrapper, or diagnostic
  use, mark it as non-default in the maintained-surface registry instead of
  leaving it ambiguous

## Manual Stage-By-Stage Execution

Follow the exact order in `project/ACTIVE_PIPELINE_FLOW.md`.

### Stage 0

Use Stage 0 only when raw Zotero-derived records or local full-text assets must
be built or refreshed.

Core scripts:

- `src/stage0_relevance/zotero_api_sync_selected.py`
- `src/stage0_relevance/zotero_fetch_llm_relevant_pdfs.py`

Completion artifact:

- `data/raw/zotero/zotero_selected_items.jsonl`

### Stage 1

Use Stage 1 when the authoritative manifest, cleaned text, or table assets are
missing or stale.

Core scripts:

- `src/stage1_cleaning/zotero_raw_to_manifest.py`
- `src/stage1_cleaning/clean_manifest_to_text.py`
- `src/stage1_cleaning/run_tables_extraction_for_dataset_v1.py`

Completion artifacts:

- `data/cleaned/index/manifest_current.tsv`
- `data/cleaned/index/key2txt.tsv`
- dataset-local cleaned text and table assets

### Stage 2

Use Stage 2 to produce the authoritative semantic-object payloads for the
declared scope.

Core script:

- `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`

Completion artifact:

- `data/results/<stage2_run_id>/semantic_stage2_objects/semantic_stage2_objects_v1.jsonl`
- supporting semantic summary and manifest sidecars

Stage2 boundary rule:

- Stage2 is the semantic discovery boundary.
- Stage2 must preserve paper-reported structure and raw expressions.
- Stage2 must not be documented as a wide-row TSV producer.
- Stage2.5 is retired from the active mainline and retained only as archived
  design history.

Authoritative Stage2 object families:

- `formulation_identity_candidate`
- `component_candidate`
- `phase_candidate`
- `process_step_candidate`
- `variable_or_factor_candidate`
- `measurement_candidate`
- `relation_cue`
- `evidence_handoff`

Compatibility bridge:

- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
- this deterministic adapter converts semantic Stage2 outputs into the legacy
  wide-row surface required by unchanged Stage3, Stage4 diagnostic, and Stage5
  runtime consumers
- the adapter is part of the active execution chain but is not a numbered
  semantic stage
- the adapter must not perform semantic inference
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
  is deprecated and retained only for fallback or debug use outside the active
  mainline

### Stage 3

Stage 3 is the deterministic formulation relation-materialization boundary.

Runtime rule:

- Stage 3 belongs to the production path
- it converts Stage 2 candidate formulation-instance rows into explicit
  paper-level relation artifacts suitable for audit and downstream closure
- it must not call any LLM or external API
- it exists to separate relation reasoning from final flattening

Current production-boundary input:

- `data/results/<stage2_run_id>/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- optional compatibility-projected JSONL and scope manifest TSV

Current production-boundary output:

- `formulation_relation_records_v1.tsv`
- `formulation_logic_graph_v1.jsonl`
- `formulation_relation_summary_v1.tsv`
- `resolved_relation_fields_v1.tsv`

Core scripts:

- `src/stage3_relation/build_formulation_relation_artifacts_v1.py`

Supporting stage-local wrapper:

- `src/stage3_relation/run_formulation_relation_artifacts_v1.py`

### Stage 4

Stage 4 is the candidate-instance diagnostic layer.

Core script:

- `src/stage4_eval/eval_weak_labels_v7pilot3.py`

Supporting review surface:

- `src/stage4_eval/build_dev15_review_workbook_v1.py`

Important:

- Stage 4 outputs are diagnostic and reviewer-facing.
- They are not the production endpoint and they are not the benchmark-valid
  system result.

### Stage 5

Stage 5 is the only active final-output and benchmark-comparison namespace.

Core scripts:

- `src/stage5_benchmark/build_minimal_final_output_v1.py`
- `src/stage5_benchmark/compare_final_table_to_gt_v1.py`

Supporting stage-local helper:

- `src/stage5_benchmark/run_minimal_final_output_v1.py`
- `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`
- `src/stage5_benchmark/build_layer2_risk_assessment_v1.py`
- `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py`
- `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`

Wrapper status:

- `src/stage5_benchmark/run_minimal_final_output_v1.py` is
  `NON-CANONICAL, STAGE5_ONLY`
- keep it only as a convenience wrapper for Stage 5A closure
- do not use it to imply hidden orchestration or complete-pipeline execution

Completion artifacts:

- `final_formulation_table_v1.tsv`
- `final_table_vs_gt_counts.tsv`
- `final_table_vs_gt_summary.md`

Required Stage 3 inputs:

- `formulation_relation_records_v1.tsv`
- `resolved_relation_fields_v1.tsv`

Optional Layer 2 GT review export:

- `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`
- This helper builds a run-scoped XLSX boundary-GT workbook from the Stage 5 final table and optional provenance artifacts.
- It is a reviewer-facing support surface, not a production-path completion artifact.

Optional Layer 2 identity scaffold binding audit:

- `src/stage5_benchmark/build_layer2_identity_scaffold_binding_v1.py`
- This helper builds a diagnostic-only scaffold-binding surface from a
  reviewed/frozen Layer2-style GT workbook plus selected Stage5 final tables.
- It applies an article-native-first binding ladder, including normalized
  namespaced-id recovery, before any strict identity-equivalent fallback.
- It is intended for pre-Layer3 value-compare validation and must not mutate
  benchmark-valid final tables or comparison outputs.
- Coarse fallback remains manual-review only and is not part of this helper's
  benchmark-grade binding surface.

Identity Freeze Gate (Mandatory)

- `src/stage5_benchmark/enforce_identity_freeze_v1.py`
- This helper validates `IDENTITY_FREEZE_RULE_V1` at the Stage5
  post-materialization boundary.
- It runs after Stage5 final-table materialization and before any:
  - value comparison
  - audit-ready export
  - Layer3 field GT evaluation
- It checks:
  - row count drift versus the upstream identity scaffold
  - identity reassignment
  - unresolved or ambiguous scaffold bindings
- It emits diagnostics and must not silently fix benchmark-valid outputs.
- Hard rule:
  - after identity freeze, downstream stages may attach, resolve, and derive
    fields only
  - downstream stages must not implicitly split or merge formulations
  - measurement fields such as size, PDI, zeta, EE, and LC must not trigger
    identity split by default
- Failure behavior:
  - if identity freeze is violated, the run is invalid and must not proceed to
    value-level evaluation
- Default behavior is enforced invariant:
  - any violation causes non-zero exit status
  - use report-only mode only for bounded diagnostics

Optional Layer 3 field GT review export:

- `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py`
- `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`
- `src/stage5_benchmark/build_value_gt_annotation_workbook_v1.py`
- `src/stage5_benchmark/run_layer3_cross_audit_v1.py`
- `src/stage5_benchmark/validate_layer3_evidence_contract_v1.py`
- These helpers build a run-scoped Layer 3 review pack from the frozen Stage 5
  final table.
- This Layer 3 pack is not only an evaluation helper.
- It is also part of the governed reviewer-facing audit and governance layer
  around the formulation database.
- The workbook may optionally consume
  `analysis/paper_risk_assessment.tsv` so Layer 2 paper-risk labels remain
  visible during field-level audit.
- The workbook may also surface article-native formulation labels, evidence-
  anchor carry-through, and normalization-needed warnings as reviewer aids.
- A separate formulation-level value GT annotation workbook may be built from
  the frozen audit-ready TSV plus the Layer 3 field seed TSV when a faster
  one-row-per-formulation numeric labeling surface is needed.
- A separate report-only Layer 3 cross-audit may be built from the compact
  value workbook plus cleaned text/tables when manual review needs one row per
  flagged cell rather than a workbook rewrite.
- Current design direction is formulation-centered:
  - reviewer entry should start from formulation rows
  - structure and identity review should come before value-credibility review
  - cell-level value-risk signals should drill down from the formulation layer
- The cross-audit helper may export deterministic rule flags plus Gemini and
  NVIDIA auditor task packs, and it may execute bounded live model audits when
  explicitly requested.
- For debugging or smoke tests, prefer bounded execution such as:
  - `--rules-only`
  - `--skip-gemini`
  - `--skip-nvidia`
  - `--max-candidates`
  - `--max-gemini-calls`
  - `--max-nvidia-calls`
  - `--batch-size`
  - `--request-timeout-seconds`
  - `--max-retries`
  - `--write-partial-every-batch`
- Model outputs remain audit signals only and must not be treated as edits or
  GT truth.
  - The workbook is not allowed to recompute weaker evidence logic than the
    active Layer 3 Evidence Handoff Contract.
- Changes to the Layer 3 workbook/export path must validate against the golden
  evidence-handoff cases before being treated as compliant reviewer-surface
  changes.
- This review pack is downstream audit support only:
  - it does not change Stage 2 extraction semantics
  - it does not change Stage 3 relation semantics
  - it does not change Stage 5 identity closure or final-table counts
  - it does not change the benchmark-valid status of
    `final_formulation_table_v1.tsv`

Path example:

- preferred:
  - `data/results/<run_id>/analysis/...`
  - `data/results/<run_id>/fgt_v3_dev15_v2/...`
- not allowed for new outputs:
  - `data/results/<run_id>/run_20260320_1317_ab12cd3_dev15_compare/...`
  - `data/results/<run_id>/compare_20260320_1317_ab12cd3/...`

Materialization rule:

- Stage 5 materializes direct extraction fields and explicit Stage 3 resolved
  relation fields only.
- Stage 5 must not perform semantic inference, donor search, or silent
  relation-layer bypass.

Production-path endpoint:

- `final_formulation_table_v1.tsv`

Comparison-node inputs:

- `final_formulation_table_v1.tsv`
- fixed manual GT workbook
- declared scope manifest TSV

Optional post-comparison audit-risk input:

- `analysis/layer2_identity_comparison.tsv` when a checked Layer 2 diagnostic
  compare surface exists for the same run scope

Optional post-comparison audit-risk outputs:

- `analysis/paper_risk_assessment.tsv`
- `analysis/paper_risk_assessment_summary.md`

Layer 2 Risk Stratification Contract:

- Purpose:
  - stratify downstream Layer 3 field-audit risk from an existing Layer 2
    identity-comparison artifact
- What it does:
  - assign deterministic paper-level `LOW` / `MEDIUM` / `HIGH` risk labels
  - assign deterministic `INCLUDE` / `REVIEW` / `HOLD` Layer 3 audit-use flags
- What it does not do:
  - it does not alter Stage 2 extraction
  - it does not alter Stage 3 relation semantics
  - it does not alter Stage 5 identity closure or benchmark-valid counts
  - it does not reinterpret the final formulation table
- Deterministic rules:
  - `LOW`: `extra_count <= 1` and `missing_count == 0`
  - `HIGH`: `extra_count > 3` or `missing_count >= 2`
  - `MEDIUM`: every non-LOW and non-HIGH paper
  - `INCLUDE` for `LOW`, `REVIEW` for `MEDIUM`, `HOLD` for `HIGH`
- Usage:
  - Layer 2 is no longer a zero-residual gate for downstream Layer 3 field
    audit
  - low-residual extra rows may be tolerated when there is no sign of
    large-scale identity collapse
  - large batch over-generation or meaningful missing identities must still be
    marked high risk and held from routine Layer 3 progression

## Evaluation Reference Path

Manual GT belongs here, not in the production path.

Reference inputs:

- `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1_fixed.xlsx`
  - preserved prior authority: `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1_fixed.xlsx`
  - current authority: `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v2_variantaware.xlsx`
- other checked manual assets under `data/cleaned/labels/manual/` when needed

Rule:

- manual GT is an external evaluation reference used only at the comparison
  node
- it is not a system-produced pipeline stage

## Run Discipline

Every `data/results/run_*` directory must include a reproducibility-grade
`RUN_CONTEXT.md`.

Required run classification:

- `intermediate_diagnostic_run`
- `component_regression_run`
- `full_pipeline_benchmark_run`

Hard rule:

- only `full_pipeline_benchmark_run` may report official GT comparison results

Corollary:

- Stage 2 outputs may be compared to GT only for diagnosis
- Stage 4 outputs may be compared to GT only for diagnosis
- the reported system result must come from Stage 5 final-table comparison only
- the fixed manual GT workbook is a reference input to the comparison node, not
  a production-stage transformation artifact

## Run-Lineage Discipline

Top-level `data/results/run_*` directories are reserved for independent
benchmark or experiment lineages.

Use a child execution under an existing lineage when the work is any of the
following:

- a retry for one or more failed papers
- a partial rerun to complete an interrupted lineage
- a deterministic repair or refresh step
- a stage-only child execution such as Stage 3 materialization for the same
  parent benchmark
- a deterministic merge or completion step that still belongs to the same
  declared lineage objective

Recommended child placement:

- `data/results/<parent_run_id>/lineage/children/<ordered_role>/<child_run_id>/`

Required parent-lineage artifacts when child runs exist:

- parent `RUN_CONTEXT.md`
- child-step mapping or index under `lineage/`
- explicit notes in the parent run flow when child paths were nested after the
  original execution

Do not create multiple sibling top-level run directories that differ only by
retry, remaining, refresh, complete, or stage suffix when they belong to the
same lineage.

## Optional Diagnostic / Review Path

The optional diagnostic/review path consists of:

- `src/stage4_eval/eval_weak_labels_v7pilot3.py`
- `src/stage4_eval/build_dev15_review_workbook_v1.py`
- `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`

This path is downstream of Stage 2 candidate extraction and exists for mismatch
analysis, review, and debugging. It is not the production endpoint.

## Incremental Reuse Discipline

Reuse is allowed only when the reused artifact is:

- unchanged
- declared in the run context
- traceable to the producing script and run

Typical valid reuse patterns:

- reuse `data/raw/zotero/zotero_selected_items.jsonl`
- reuse `data/cleaned/index/manifest_current.tsv`
- reuse cleaned text or table assets
- reuse a Stage 3 relation-record TSV while iterating only on Stage 5 closure
- reuse a Stage 2 candidate TSV while iterating on Stage 5 closure logic
- reuse the fixed manual GT workbook as a declared comparison input

Typical invalid reuse patterns:

- using an undocumented hand-edited TSV as a stage completion artifact
- mixing artifacts from different declared scopes
- skipping Stage 5 comparison and claiming benchmark validity anyway

## Current Canonical Endpoints

Final structured formulation output:

- `final_formulation_table_v1.tsv`

Comparison-node outputs:

- `final_table_vs_gt_counts.tsv`
- `final_table_vs_gt_summary.md`

The production path yields the final formulation table. The benchmark-valid
result is obtained only when the comparison node reads that table together with
the fixed GT workbook and declared scope manifest as separate inputs.

## What This Runbook Does Not Allow

- no hidden one-click full rerun wrapper
- no duplicate active Stage 5 namespace
- no reporting of Stage 2 or Stage 4 artifacts as final benchmark outputs
- no undocumented shortcut from partial artifacts directly to benchmark claims

## Layer 1 GT Counting Rule

Layer 1 GT counts are formulation-instance counts, not full design-space
counts.

- Count a GT formulation only when the paper reports a realized experimental
  instance with row-level or result-level evidence.
- Do not count methods-only combinations or sweep conditions that were not
  reported as concrete formulation instances.
