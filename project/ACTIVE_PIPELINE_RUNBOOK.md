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
  - The 2026-03-30 corrective architecture freeze restores the original split:
    - LLM owns open semantic discovery and formulation-boundary discovery in Stage2
    - deterministic post-LLM completion remains inside Stage2 for downstream readiness
    - deterministic logic in Stage3+ owns relation resolution, inheritance,
      normalization, filtering, audit, and materialization
- Deterministic semantic emitters and semantic lifts are fallback,
  comparator, migration-support, or diagnostic infrastructure only.
- The deterministic compatibility adapter remains deterministic downstream
  infrastructure; it does not own Stage2 semantic authority.
- Current direction note:
  - the restored semantic-authority boundary remains correct and in force
  - the remaining bottleneck is semantic contract overload, not permission to
    re-promote deterministic semantic authority
  - future contract work should reduce LLM output burden and shift more
    execution-level strictness into governed downstream function units and
    validators where that can be done without losing semantic authority
- Non-change reminder:
  - Stage 5 remains materialization-only.
  - The relation-first Stage 3 -> Stage 5 contract is unchanged.
- Legacy note:
  - The deprecated wide-row fallback extractor and the deterministic semantic
    emitter may still carry useful fallback/comparator behavior, but neither is
    the frozen Stage2 authority contract.

## Operating Principle

A complete pipeline run means the declared scope has passed through every
required production stage from upstream corpus inputs to:

- `final_formulation_table_v1.tsv`

Benchmark-valid reporting then additionally requires the comparison node, which
reads:

- the final formulation table
- the frozen Layer1 GT counts TSV
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
- reuse unchanged completed Stage 2 artifacts

Not allowed:

- skip a required stage entirely
- substitute undocumented files for canonical stage completion artifacts
- report a partial-layer artifact as if it were the final system result

Run layout rule:

- future governed lineage root:
  - `data/results/<YYYYMMDD_<short_hash>>/`
- future child execution root:
  - `data/results/<YYYYMMDD_<short_hash>>/<NN_<cue>>/`
- future child execution roots must carry rich execution meaning in
  `RUN_CONTEXT.md`, not in long folder names
- artifact folders under a child execution root must be functional only, such as:
  - `analysis/`
  - `outputs/`
  - `audit/`
  - stage-local functional folders like `formulation_relation_v1/`
- future lineage must not repeat a full nested `run_id` or timestamp/hash
  fragment below the governed bucket root
- historical `run_*` roots remain legacy compatibility surfaces
- maintained writers that create new run surfaces now default to the MDEC084
  v2 bucket/child layout
- explicit legacy `run_*` creation remains compatibility-only and must be
  requested explicitly

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

Use Stage 2 to produce the authoritative completed Stage2 payloads for the
declared scope.

Frozen authority contract:

- Stage2 authority belongs to one composite stage:
  - internal LLM semantic-object discovery
  - internal deterministic post-LLM completion
- Deterministic semantic reconstruction must not be selected as active mainline
  authority outside that governed composite Stage2 path.
- No formulation candidate may enter the authoritative completed Stage2 artifact
  unless it is traceable to:
  - `llm_semantic_discovery`
  - or an explicitly declared governed fallback mode
- Deterministic completion may expand, enumerate rows, normalize, repair,
  project, and bridge only within the authorized semantic scope for the run.
- In `llm_first_composite` mode, deterministic DOE row expansion is lawful only
  within an LLM-declared DOE scope.

Current clarification on remaining Stage2 pressure:

- The current maintained Stage2 contract is considered healthy on authority:
  - no deterministic semantic overreach is restored
  - marker provenance remains governed
  - deterministic execution remains scope-authorized
  - the maintained validator remains part of the guardrail surface
- The remaining issue is that the live semantic contract may still push the
  LLM toward candidate-level or partially execution-ready structure too early.
- Future contract redesign should therefore target governed intermediate
  semantic markers and reusable semantic cues rather than simply tightening the
  current prompt for more execution-ready certainty.
- The maintained runtime now includes one narrow contract relaxation:
  - `selection_marker` and `inheritance_marker` may be preserved as governed
    `partial_semantic` markers when only the currently identified
    non-execution-critical grounding fields are incomplete
  - downstream deterministic execution remains restricted to
    `execution_ready` markers
- This does not authorize deterministic semantic inference.

Current implementation-status note:

- `src/stage2_sampling_labels/run_stage2_composite_v1.py` is the one governed
  Stage2 execution entrypoint.
- The governed Stage2 wrapper refreshes run-level feature activation observability after writing `RUN_CONTEXT.md`.
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py` is the
  internal LLM semantic-discovery substep used by the governed Stage2 entrypoint.
- The same maintained Stage2 path may replay saved raw responses without new
  LLM calls and rehydrate current live-v2 raw-response freezes back into the
  authoritative completed Stage2 artifact.
- The maintained Stage2 path should be read through the following internal
  governance mapping:
  - `S2-1 Scope resolution`
  - `S2-2 Evidence construction`
  - `S2-3 Prompt assembly`
  - `S2-4 LLM call`
  - `S2-5 Semantic parsing`
  - `S2-6 Contract validation`
  - `S2-7 Compatibility projection`
- The maintained Stage2 path now formalizes an internal S2-2 boundary:
  - clean text -> governed evidence package
  - explicit internal sub-boundary:
    `clean text / extracted tables -> candidate segmentation -> role-aware selector -> governed evidence package`
  - candidate artifact:
    `semantic_stage2_objects/candidate_blocks/<paper_key>/candidate_blocks_v1.json`
  - canonical artifact:
    `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
  - S2-2 is the first engineering freeze point in Stage2
  - producer:
    `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  - consumer:
    the same maintained extractor's role-aware selector consumes candidate
    blocks and its prompt-assembly path consumes the canonical evidence
    artifact
  - candidate segmentation is responsible for structure recovery,
    conservative section-aware splitting, table isolation, and conservative
    noise filtering only
  - `analysis/stage2_prompt_preview_v1.tsv` remains maintained observability,
    but it is derived from the canonical evidence artifact and is not the
    primary truth surface
  - `analysis/candidate_segmentation_debug_v1.tsv` is the maintained
    candidate-level observability surface before selector prioritization
  - the maintained selector for this boundary is deterministic and role-aware:
    - default general profile:
      `PREPARATION_METHOD`, `MATERIALS`, `FORMULATION_TABLE`,
      `FORMULATION_RESULT`, `OPTIMIZATION_RESULT`, `CONTEXT_FALLBACK`
    - DOE optimization overlay:
      `PREPARATION_METHOD`, `MATERIALS`, `EXPERIMENTAL_DESIGN`,
      `VARIABLE_TABLE`, `FORMULATION_TABLE`, `OPTIMIZATION_RESULT`,
      `CONTEXT_FALLBACK`
    - no second LLM is used for pre-LLM evidence selection
  - role selection is constrained by role coverage rather than pure global
    top-K ranking, and duplicate near-identical tables may be suppressed
  - the canonical artifact records `selector_profile`,
    `archetype_overlay`, per-block role fields, and any weak or missing roles
  - the artifact records separate `technical_status` and `design_status`
  - if the artifact is readable but the intended input-contract path was not
    satisfied, that must remain visible as design nonconformance rather than
    silent success
- S2-3 prompt assembly may consume `evidence_blocks_v1.json` only:
  - it must not reread clean text
  - it must not perform new selection or ranking
- S2-4 is the only nondeterministic Stage2 substep and emits raw LLM response
  payloads for live or replay-backed processing
- S2-5 parses raw LLM responses into Stage2 semantic-object artifacts
- Stage2 segmentation closure freeze rule:
  once S2-2a segmentation closure is declared for the current cycle,
  candidate-segmentation logic is frozen by default
- post-closure operating rule:
  remaining S2-2 closure work should target selector/evidence prioritization
  or table-extraction-quality diagnosis first, and segmentation changes require
  concrete regression evidence against the frozen closure state
- S2-2b strict stage-local debugging rule:
  - optimize one stage and freeze one stage
  - judge S2-2b only inside the S2-2 boundary on frozen S2-2a inputs
  - selector must not introduce new candidate discovery behavior and operates
    strictly on existing `candidate_blocks_v1.json`
  - do not use S2-3 prompt assembly, S2-4 LLM call, S2-5 semantic parsing,
    S2-6 contract validation, or S2-7 compatibility projection to decide
    selector closure
  - do not use any Stage3, Stage4, Stage5, or GT-comparison signal for S2-2b
    debugging or closure
  - allowed audit surfaces only:
    - `semantic_stage2_objects/candidate_blocks/<paper_key>/candidate_blocks_v1.json`
    - `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
    - `analysis/table_selection_debug_v1.json`
    - `analysis/candidate_segmentation_debug_v1.tsv`
    - stage-local selector audit against human reference passages
  - allowed labels only:
    - `selector_miss`
    - `evidence_ranking_miss`
    - `wrong_table_choice`
    - `selected_table_unusable_as_non_target`
  - explicit audit loop:
    - build a paper-level audit on the frozen S2-2a inputs
    - classify each paper with one allowed S2-2b label
    - implement the smallest selector-only fix
    - rerun on the same frozen inputs
    - repeat until closure or until remaining issues are non-target
  - execution discipline:
    - each micro-step must be independently auditable from the allowed S2-2
      artifacts
    - no cross-stage reasoning is allowed in the S2-2b audit
    - no downstream signal may be used to upgrade or downgrade a selector
      judgment
  - closure means acceptable selector behavior on frozen S2-2a inputs
    relative to the human reference within S2-2
  - closure criteria:
    - no major `selector_miss` remains in the target set
    - no major `wrong_table_choice` remains in the target set
    - `evidence_ranking_miss` is no longer the dominant residual failure mode
    - remaining residuals are mostly
      `selected_table_unusable_as_non_target`
    - selector behavior is acceptable relative to the human reference passages
      at the S2-2 boundary
    - the result is reproducible across at least one repeated S2-2-local run
      under the same configuration
    - closure must be recorded explicitly as stage-local selector freeze only
  - this is a stage-local selector freeze only, not downstream system
    validation
## S2-2b Human Reference Passages (Audit Anchor)
  S2-2b human reference passages are sourced from:

  ```text
  docs/selector_calibration/
  ```

  This directory is treated as a **frozen, authoritative audit anchor** for S2-2b selector evaluation.

  Rules:

  * These passages represent the intended formulation-relevant evidence as determined by human calibration.
  * They are used ONLY for stage-local selector auditing.
  * They MUST NOT be modified during S2-2b iterations.
  * They MUST NOT be regenerated or extended based on selector behavior.
  * They MUST NOT be used for downstream stages (Stage3/Stage4/Stage5).
  * They MUST NOT be treated as GT or benchmark targets.

  If multiple files exist in the directory:

  * treat the entire directory as a fixed reference set
  * do NOT perform file selection heuristics
  * assume all content is relevant to the calibration scope
- Stage2 live input assembly can be switched to the governed ordered-evidence-pack mode by setting `STAGE2_INPUT_EVIDENCE_PACKING_MODE=ordered_blocks`; the default remains the current raw-prefix path.
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py` is
  the internal deterministic post-LLM completion substep used by the governed
  Stage2 entrypoint.
- That projection step is S2-7:
  - compatibility projection and Stage3 handoff
  - not evidence construction
- `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py`
  is the maintained contract check invoked by the governed Stage2 entrypoint.
  It fails when semantic-source mode is mixed or undeclared, when row
  provenance is missing, or when deterministic DOE expansion appears without
  LLM-declared DOE scope in `llm_first_composite` mode.
- That validator is S2-6:
  - contract validation, not selector logic
- `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
  remains available only for explicitly declared fallback, comparator,
  migration-support, or diagnostic use and must not be treated as the governed
  Stage2 path.
- `src/utils/run_threepaper_stage2_v2_comparison.py` is a governed,
  supporting-nondefault wrapper for a three-paper Stage2 semantic-intermediate
  comparison slice. It is allowed for architecture-enforcement and same-scope
  comparison only and must not be treated as a promotion of Stage2 authority or
  as a replacement for the canonical Stage0-Stage5 runbook.
- Downstream validation note:
  runs that traverse later Stage2 substeps under the maintained entrypoint are
  separate later tasks and must not be used for S2-2b debugging or closure
  judgment.

Boundary legality note:

- The active pipeline distinguishes `internal_intermediate`,
  `diagnostic_boundary`, `mainline_resume_boundary`, and
  `benchmark_terminal_boundary`.
- A replayable artifact is not automatically a lawful resume boundary; it must
  preserve the authoritative downstream-ready contract for the next stage.
- Raw Stage2 freeze baselines are diagnostic boundaries unless they contain the
  completed Stage2 weak-label artifact required by Stage3.
- A frozen current live-v2 raw-response set may be rehydrated into that lawful
  Stage2 boundary only by replaying it through the maintained composite Stage2
  entrypoint and maintained internal completion step.
- The completed Stage2 weak-label TSV is the lawful resume boundary into
  Stage3.
- The Stage5 final formulation table plus comparison outputs are the benchmark
  terminal boundary.

Completion artifact:

- `data/results/<stage2_run_id>/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- supporting completed-Stage2 JSONL and projection trace sidecars

Maintained replay/rehydration rule:

- When `run_stage2_composite_v1.py` is invoked with `--source-mode
  legacy_llm_replay`, the maintained extractor may consume:
  - historical legacy raw-response payloads
  - current live-v2 raw-response payloads already saved under
    `semantic_stage2_objects/raw_responses/`
- Replay mode remains inside Stage2.
- The replayable raw-response directory is not itself authoritative; the
  authoritative downstream boundary is still the completed Stage2 artifact
  emitted by the same maintained composite Stage2 path.

Internal Stage2 intermediate:

- `data/results/<stage2_run_id>/semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
- `data/results/<stage2_run_id>/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
- supporting semantic summary and raw-response sidecars

Stage2 boundary rule:

- Stage2 is a composite stage, not a new numbered stage family.
- Stage2 owns semantic discovery and deterministic post-LLM completion.
- Stage2 must preserve paper-reported structure and raw expressions.
- The raw LLM semantic-object payload is an internal intermediate.
- The completed Stage2 artifact is the only valid Stage3 input and the only
  authoritative Stage2 evaluation target.
- Deterministic semantic emitters, deterministic semantic lifts, and similar
  reconstruction paths are not allowed to stand in as Stage2 authority.
- Maintained Stage2 runs must declare exactly one semantic-source mode:
  - `llm_first_composite`
  - `governed_fallback_semantic_source`
  - `diagnostic_comparator`
- Completed Stage2 rows must preserve additive provenance fields that answer:
  - who declared the candidate universe
  - who materialized the row
  - what semantic scope authorized the row to exist
- Stage2.5 is retired from the active mainline and retained only as archived
  design history.

Required boundary fields for governed resume-capable runs:

- `boundary_class`
- `authoritative_for_downstream`
- `lawful_resume_boundary`
- `resume_entrypoint`
- `schema_contract`
- `upstream_authority_source`
- `replay_mode`

These fields are additive run-context metadata only. They do not change stage
authority or promote diagnostic boundaries into mainline resume boundaries by
themselves.
The maintained RUN_CONTEXT refresher
`src/utils/update_run_context_with_feature_activation_v1.py` now injects these
boundary-governance fields automatically for run-producing maintained wrappers
that call it after their primary `RUN_CONTEXT.md` write.

Authoritative Stage2 object families:

- `formulation_identity_candidate`
- `component_candidate`
- `phase_candidate`
- `process_step_candidate`
- `variable_or_factor_candidate`
- `measurement_candidate`
- `relation_cue`
- `evidence_handoff`

Deterministic post-LLM completion:

- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
- this deterministic adapter converts the Stage2 semantic intermediate into the
  completed Stage2 artifact required by unchanged Stage3, Stage4 diagnostic,
  and Stage5 runtime consumers
- the adapter is part of the active execution chain and remains inside Stage2
- the adapter must not perform semantic inference
- the adapter now filters governed partial semantic selection/inheritance
  markers out of the execution-facing row handshake so unchanged downstream
  consumers continue to see only execution-ready markers
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
- The Stage 3 wrapper also refreshes run-level feature activation observability after writing `RUN_CONTEXT.md`.

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

The compare entrypoint also refreshes run-level feature activation observability
after writing `RUN_CONTEXT.md`.

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
- The Stage 5 wrapper also refreshes run-level feature activation observability after writing `RUN_CONTEXT.md`.

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
- Confirmed lineage-specific workbook-role note:
  - in `run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1`, the operative human-reviewed Layer2 boundary decision workbook is
    `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/boundary_gt_review_v1/boundary_gt_review_workbook_v1.xlsx`
  - use that workbook as the practical base for future manual rehydration-vs-GT boundary comparison work in that lineage
  - describe it as a run-scoped reviewed boundary surface seeded from Stage5 final rows, not as the repository-wide ultimate raw GT origin

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
- Confirmed lineage-specific workbook-role note:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4_with_pH.xlsx`
    is a downstream field/value annotation workbook
  - it inherits the accepted formulation universe from the reviewed
    `include_gt` subset of the boundary workbook above
  - do not use it as the base for new Layer2 boundary comparison or
    rehydration-vs-GT diff work
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
- frozen Layer1 GT counts TSV
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

Every future governed child execution directory under
`data/results/<YYYYMMDD_<short_hash>>/<NN_<cue>>/` must include a
reproducibility-grade `RUN_CONTEXT.md` that records both script lineage and
feature activation lineage.

Historical maintained `data/results/run_*` directories remain legacy
compatibility surfaces and should also continue to carry `RUN_CONTEXT.md`.

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
- `RUN_CONTEXT.md` is incomplete for governed runs if it does not include the
  generated `## Feature Unit Activation` section.

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

- future default:
  - `data/results/<YYYYMMDD_<short_hash>>/<NN_<cue>>/`
- historical legacy lineages may still appear under older nested `run_*`
  paths until an explicit migration is approved

Required parent-lineage artifacts when child runs exist:

- bucket-level lineage note or index describing the ordered child executions
- child `RUN_CONTEXT.md` files for each governed execution folder
- explicit notes when historical legacy nested paths are still being referenced

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
the frozen Layer1 GT counts TSV and declared scope manifest as separate inputs.

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
