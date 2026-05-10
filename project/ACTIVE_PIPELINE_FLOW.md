# Active Pipeline Flow

This document is the single authoritative manual reproduction guide for the
current canonical pipeline.

It defines one explicit production path that starts from Zotero-derived raw
records and ends at:

- final per-formulation structured records

It also defines the separate evaluation reference inputs and the final
comparison node that reads the production output against manual GT.

This file is intentionally operational. It is the document a human should follow
when reproducing the canonical path step by step without relying on any hidden
Python orchestration.

## Purpose And Scope

The canonical pipeline exists to produce auditable formulation-level outputs
with full provenance from upstream corpus inputs through final formulation-table
production.

Current scope:

- corpus intake and relevance filtering from Zotero-derived raw records
- manifest and cleaned-content construction
- Stage 2 composite semantic extraction:
  - LLM semantic discovery
  - deterministic post-LLM completion into the downstream-ready Stage2 artifact
- Stage 3 deterministic formulation relation materialization layer
- Stage 4 candidate-instance diagnostics and review surfaces
- Stage 5 final formulation-table closure
- a separate Stage 5 comparison node that reads the final formulation table and
  fixed manual GT workbook as separate inputs

Corrective architecture-freeze note (`2026-03-30`):

- The Stage2 authority-transition audit confirmed that deterministic semantic
  Stage2 authority was an architecture drift rather than a cleanly closed
  replacement of the earlier LLM-centered Stage2 contract.
- The frozen Stage2 authority contract is:
  - LLM for open semantic discovery and formulation-boundary discovery
  - deterministic post-LLM completion inside Stage2 for downstream readiness
  - deterministic downstream logic in Stage3+ for relation resolution,
    inheritance handling, normalization, filtering, audit, and materialization
- Deterministic semantic emitters and semantic lifts remain available only for
  fallback, comparator, migration-support, or diagnostic work.
- They must not be treated as active Stage2 mainline authority.

Current clarification (`2026-04-08`):

- The architecture correction successfully restored semantic authority.
- The remaining failure mode is not a return to deterministic semantic
  overreach.
- The remaining failure mode is semantic contract overload inside the current
  Stage2 semantic substep:
  - the LLM is still pressured toward candidate-universe construction and
    partially execution-ready structure emission earlier than preferred
  - strict marker requirements can suppress governed marker families when the
    paper-level semantic pattern is understood but exact execution-level
    grounding is incomplete
- This clarification does not change the active runtime contract yet.
- It records the preferred future direction for contract redesign:
  - reduce LLM output burden
  - allow governed partial semantic markers where appropriate
  - keep deterministic execution strictness downstream

The canonical benchmark object is the Stage 5 final formulation table. No
intermediate artifact may be reported as the system result against GT.

Current Phase: Diagnostic Development Mode

- The pipeline is currently in diagnostic development mode.
- Stage5 outputs are diagnostic baselines for debugging work.
- Benchmark mode is disabled in the current phase.
- Any reference to a baseline in this phase means diagnostic baseline unless a governed contract explicitly narrows it further.

Benchmark-legality clarification:

- Stage5 final-table materialization is necessary but not sufficient for a
  benchmark-valid run.
- The GT compare node remains separate and consumes only the completed Stage5
  final table, declared scope manifest, and frozen GT authority.
- A full DEV15 lineage can therefore reach Stage5 final-table materialization
  and remain diagnostic-only while benchmark mode is disabled.

Fine-grained governance mapping:

- Stage0 / Stage1:
  - `S1-1 Raw ingestion`
  - `S1-2 Multi-source manifest assembly`
  - `S1-3 Manifest hydration`
  - `S1-3a Asset hydration`
  - `S1-3b Scope overlays`
- Stage2:
  - `S2-1 Scope resolution`
  - `S2-1b High-confidence source denoise projection`
  - `S2-2 Evidence construction`
  - `S2-2a Candidate segmentation`
  - `S2-2b Selector evidence prioritization`
  - `S2-3 Prompt assembly`
  - `S2-4a Prompt construction freeze boundary`
  - `S2-4b Live LLM call freeze boundary`
  - `S2-5 Semantic parsing`
  - `S2-6 Contract validation`
  - `S2-7 Compatibility projection`
- Stage3:
  - `S3-1 Relation materialization`
  - `S3-2 Relation resolution`
- Stage5:
  - `S5-1 Fixed-row candidate intake`
  - `S5-2 Deterministic direct materialization`
  - `S5-3 LLM-assisted direct value candidate extraction`
  - `S5-4 Value authority validation and merge`
  - `S5-5 Derived reasoning / calculated value materialization`
  - `S5-6 Final table closure and audit export`
  - S5-3 is residual source-evidenced gap filling, not database completion;
    heterogeneous literature reporting must not be coerced into dense
    row-by-field extraction merely because schema fields are blank.
- Benchmark:
  - `B-1 GT compare`
- Cross-cutting layers:
  - Feature governance layer
  - Memory layer

This mapping is internal governance only. It does not create new coarse
runtime stages or new runtime namespaces.

Boundary classes used by the current maintained pipeline:

- `internal_intermediate`
  - a stage-local artifact that supports execution but is not itself a resume
    boundary
- `diagnostic_boundary`
  - a replay or audit artifact that can be inspected for debugging but does
    not by itself authorize mainline continuation
- `mainline_resume_boundary`
  - a contract-complete upstream artifact that the maintained downstream stage
    may legally consume without boundary drift
- `benchmark_terminal_boundary`
  - the final governed benchmark surface for the active pipeline

Boundary rule:

- raw Stage2 semantic objects and raw freeze baselines are diagnostic
  boundaries unless they also preserve the completed Stage2 authority surface
  required by Stage3
- a frozen current live-v2 raw-response set may be converted into the
  authoritative completed Stage2 surface only by replaying it through the
  maintained composite Stage2 entrypoint and its maintained internal Stage2
  completion step; the raw freeze itself does not become a mainline resume
  boundary until that completion artifact is produced
- the completed Stage2 weak-label TSV is the lawful downstream resume boundary
  for Stage3
- the Stage5 final formulation table and comparison outputs are the benchmark
  terminal boundary

## Canonical Path Definition

The system canonical production path is:

1. Stage 0: build or refresh Zotero-derived raw records and local source assets
2. Stage 1: convert raw records into the authoritative manifest, cleaned text,
   and table assets
3. Stage 2: run the governed composite Stage2 graph:
   - LLM semantic discovery from cleaned assets
   - deterministic post-LLM completion into the downstream-ready Stage2 artifact
4. Stage 3: materialize explicit paper-level formulation relation artifacts
   from the completed Stage2 artifact
5. Stage 4: optionally generate candidate-level diagnostic and reviewer-facing
   artifacts
6. Stage 5: close candidate rows into final formulation rows with optional
   Stage 3 relation provenance, deterministic direct value materialization,
   optional LLM-assisted direct value candidates, authority validation, and
   separately-provenanced derived value sidecars

Active-transition note:

- Stage2.5 is retired from the active mainline and remains archived only as a
  historical exploratory side-path.
- The frozen architecture contract assigns Stage2 semantic authority to an LLM
  semantic-discovery boundary.
- The deterministic compatibility adapter is an internal Stage2 completion
  substep; it does not own Stage2 semantic authority and it is not a separate
  numbered stage.
- The deterministic semantic emitter and the legacy wide-row extractor are both
  retained only for fallback, comparator, migration-support, or debug use.
- The governed three-paper Stage2 v2 comparison slice under
  `src/utils/run_threepaper_stage2_v2_comparison.py` is a non-mainline
  semantic-intermediate comparator path only. It may emit Stage2 semantic-
  intermediate artifacts and comparison packs for `WIVUCMYG`, `UFXX9WXE`, and
  `5GIF3D8W`, but it does not promote Stage2 authority, define Stage2 itself,
  or alter the canonical Stage0-Stage5 path.

The production-path completion artifact is:

- `final_formulation_table_v1.tsv`

The evaluation reference path is separate:

- fixed manual GT assets under `data/cleaned/labels/manual/`
- declared scope manifest TSVs

The final comparison node is separate from production:

- input A: `final_formulation_table_v1.tsv`
- input B: frozen Layer1 GT counts TSV derived from the locked DEV15 GT authority
- input C: declared scope manifest TSV
- output: `final_table_vs_gt_counts.tsv` and `final_table_vs_gt_summary.md`
  - `analysis/paper_risk_assessment_summary.md`

The active stage namespaces are exactly:

- `src/stage0_relevance/`
- `src/stage1_cleaning/`
- `src/stage2_sampling_labels/`
- `src/stage3_relation/`
- `src/stage4_eval/`
- `src/stage5_benchmark/`

Reserved reference namespace:

- `src/stage3_gt/`

There is no active Stage 6 and no active Stage 7. There is also no active
second Stage 5 namespace.

For execution-facing benchmark, comparison, workbook, and audit workflows, use
only the maintained entrypoints declared in
`project/ACTIVE_PIPELINE_RUNBOOK.md` and `docs/maintained_script_surface.tsv`.
Do not choose scripts by filename similarity or branch-era convenience.

For canonical provenance, the pipeline starts from raw Zotero-derived records
under `data/raw/zotero/`. In current repo practice, the checked Stage 0
completion artifact is `data/raw/zotero/zotero_selected_items.jsonl`, whether it
was produced from a Zotero export workflow or from the checked sync script.

## Stage-By-Stage Flow

### Stage 0. Relevance Filtering And Raw Corpus Intake

Purpose:
Build the raw Zotero-derived item set and local full-text assets that later
stages depend on.

Exact input files or directories:

- `data/raw/zotero/`
- Zotero-derived raw item selection, export, or sync source
- local Zotero storage root for attachments when using sync/fetch scripts

Exact script path(s) and script filename(s):

- `src/stage0_relevance/zotero_api_sync_selected.py`
- `src/stage0_relevance/prefilter_regex.py`
- `src/stage0_relevance/classify_gemini_grouped.py`
- `src/stage0_relevance/zotero_fetch_llm_relevant_pdfs.py`

Exact output files or directories:

- `data/raw/zotero/zotero_selected_items.jsonl`
- `data/raw/zotero/zotero_llm_relevant.jsonl`
- local PDF or HTML attachment paths resolved into the raw Zotero records

Stage completion artifact:

- `data/raw/zotero/zotero_selected_items.jsonl`

Consumed by downstream stage:

- Stage 1

### Stage 1. Manifest, Clean Text, And Table Assets

Purpose:
Turn the raw Zotero-derived item set into stable cleaned assets and the
authoritative manifest used by extraction.

Exact input files or directories:

- one or more declared raw Zotero JSONL artifacts such as:
  - `data/raw/zotero/zotero_selected_items.jsonl`
  - `data/raw/zotero/zotero_collection__goren_2025.jsonl`
- local PDF and HTML files referenced by the raw records

Exact script path(s) and script filename(s):

- `src/stage1_cleaning/zotero_raw_to_manifest.py`
- `src/stage1_cleaning/hydrate_manifest_v1.py`
- `src/stage1_cleaning/clean_manifest_to_text.py`
- `src/stage1_cleaning/run_tables_extraction_for_dataset_v1.py`

Exact output files or directories:

- `data/cleaned/index/manifest_current.tsv`
- `data/cleaned/index/key2txt.tsv`
- `data/cleaned/content/text/`
- dataset-local cleaned content roots such as `data/cleaned/goren_2025/`
- dataset-local table assets such as `data/cleaned/goren_2025/tables/`

Stage completion artifact:

- the manifest and cleaned text mapping:
  - `data/cleaned/index/manifest_current.tsv`
  - `data/cleaned/index/key2txt.tsv`

Consumed by downstream stage:

- Stage 2

### Stage 2. Composite Semantic Extraction And Completion

Purpose:
Run the governed composite Stage2 path on cleaned paper content and tables.
Stage2 contains:

1. deterministic scope resolution and high-confidence source-denoise projection before evidence selection
2. LLM semantic discovery
3. deterministic post-LLM completion into the downstream-ready Stage2 artifact

The raw semantic-object payload is an internal Stage2 intermediate, not the
authoritative Stage2 completion artifact.

Current design-direction clarification:

- Stage2 has already restored the rule that LLM semantic discovery is the
  semantic authority.
- The remaining bottleneck is that the current semantic contract can still ask
  the LLM for more execution-like structure than is ideal.
- The preferred future Stage2 contract direction is for the LLM to emit
  reusable semantic cues and governed intermediate markers, not final
  executable formulation structures.
- This does not authorize uncontrolled vague text and does not authorize
  deterministic semantic inference.
- It means that when the paper supports a governed semantic cue but not full
  execution-ready grounding, the future contract direction should prefer
  governed intermediate expression over silent suppression where governance can
  keep the downstream behavior auditable.

Current maintained contract note:

- the maintained Stage2 prompt and validator now allow governed
  `partial_semantic` markers for:
  - `selection_marker`
  - `inheritance_marker`
- this relaxation is limited to the currently identified non-execution-
  critical grounding fields:
  - `selection_marker.source_table_id`
  - `selection_marker.selected_variable`
  - `selection_marker.selected_value`
  - `inheritance_marker.from_table`
  - `inheritance_marker.to_table`
- strict execution-critical fields remain unchanged in the active runtime,
  especially:
  - `inheritance_marker.inherit_type`
  - `inheritance_marker.variable`
  - `inheritance_marker.value`
- partial markers are preserved in the Stage2 semantic-intermediate artifact
  but do not by themselves authorize current deterministic execution
- Stage2 decomposition also exposed a real execution-ownership failure:
  semantic signals could exist while governed deterministic function units were
  not reliably taking control on the active mainline.
- Silent non-activation is not acceptable when semantic authorization is
  present.
- Current governed status:
  - DOE execution is restored on-path for `UFXX9WXE` in
    `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix`, where governed
    deterministic execution emitted `26` DOE rows
  - non-DOE table-row execution has partial downstream repair only
  - the dominant remaining blocker for broader DEV15 coverage is upstream
    missing `table_formulation_scopes` in the Stage2 extraction or selector or
    evidence-handoff path

Exact input files or directories:

- cleaned manifest or split manifest TSVs
- raw/current cleaned text assets resolved by S2-1
- S2-1b denoised-for-Stage2 text projections when present or generated for the run
- cleaned table assets when available

Exact script path(s) and script filename(s):

- `src/stage2_sampling_labels/run_stage2_composite_v1.py`

Operational execution note:

- `run_stage2_composite_v1.py` is the governed coarse-grained Stage2 wrapper
  for the full composite contract and the lawful replay/rehydration path that
  can traverse through `S2-5`, `S2-6`, and `S2-7` into the completed Stage2
  artifact.
- Current maintained Stage2 execution also includes dedicated fine-grained
  execution-facing surfaces for frozen substeps where governance has made
  them explicit, including:
  - `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py`
  - `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`
  - `src/stage2_sampling_labels/run_stage2_s2_5_semantic_parsing_v1.py`
  - `src/stage2_sampling_labels/run_stage2_s2_6_contract_validation_v1.py`
  - `src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py`
- These dedicated runners do not replace the composite Stage2 contract or the
  lawful replay/rehydration requirement for authoritative completed Stage2
  output.

Internal Stage2 scripts:

- planned high-confidence source-denoise projection substep:
  - `src/stage2_sampling_labels/denoise_stage2_source_text_s2_1b_v1.py`
- semantic extraction substep:
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
- deterministic post-LLM completion substep:
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`

Comparator / fallback note:

- `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
  remains available only for fallback, comparator, migration-support, or
  diagnostic runs and must not be treated as Stage2 authority.
- Governed Stage2 runs must declare exactly one semantic-source mode:
  - `llm_first_composite`
  - `governed_fallback_semantic_source`
  - `diagnostic_comparator`
- In `llm_first_composite` mode, deterministic DOE row enumeration may
  materialize row-level candidates only within LLM-declared DOE scope.
- In `llm_first_composite` mode, deterministic non-DOE table row enumeration
  may materialize row-level candidates only when the LLM declares a
  formulation-bearing non-DOE table through the table authorization contract.
- within that non-DOE path, a bounded simple-table enumerator may materialize
  rows directly from preserved `S2-2` normalized payload authority when the
  authorized table is a low-ambiguity `full_formulation` surface with stable
  first-column row identity
- this bounded simple-table enumerator does not require LLM row-level output
  and must not take over DOE matrices, sweep-family recovery, or cross-table
  decoding
- `src/stage2_sampling_labels/extract_semantic_stage2_v2_threepaper.py`
  is a compatibility shim only and must not be treated as the Stage2 definition.

Stage2 -> deterministic -> Stage3 handshake:

- the LLM substep may declare table-level semantic markers only:
  - `table_formulation_scope`
  - `variable_roles`
  - `selection_marker`
  - `inheritance_marker`
  - `boundary_marker`
- the deterministic completion substep may then choose one of two governed row
  expansion paths:
  - DOE enumerator for `is_doe == true`
  - table row expansion for `is_formulation_table == true` and `is_doe == false`
- the table-row expansion path may then choose among bounded internal
  strategies:
  - simple-table deterministic enumeration for low-ambiguity
    `full_formulation` tables
  - existing non-DOE table-row recovery logic for more complex preserved
    table surfaces
- both expansion paths must emit compatible candidate-row schemas for Stage3
  relation binding.

Handshake clarification:

- the current maintained pipeline still uses some candidate-ready semantic
  structures inside the Stage2 semantic intermediate
- the current maintained pipeline now also distinguishes marker readiness:
  - `execution_ready`
  - `partial_semantic`
- only `execution_ready` markers are handed to the current deterministic
  execution path through the completed Stage2 row surface
- future contract evolution should move this handshake toward governed
  intermediate semantic protocol behavior where:
  - the LLM contributes semantic cues and authorization markers
  - deterministic function units perform more of the execution-level
    completion
- this is a future-facing contract direction only and must not be described as
  already implemented runtime behavior

Exact output files or directories:

- run-scoped semantic-object outputs under
  `data/results/run_<run_id>/semantic_stage2_objects/`
- canonical S2 internal candidate-segmentation artifacts under
  `data/results/run_<run_id>/semantic_stage2_objects/candidate_blocks/<paper_key>/`
- canonical S2-2 pre-LLM evidence artifacts under
  `data/results/run_<run_id>/semantic_stage2_objects/evidence_blocks/<paper_key>/`
- canonical semantic-object artifacts:
  - `semantic_stage2_objects/candidate_blocks/<paper_key>/candidate_blocks_v1.json`
  - `semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`
  - `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
  - `semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
  - `semantic_stage2_objects/semantic_stage2_v2_summary.tsv`
  - `semantic_stage2_objects/raw_responses/`
  - `analysis/candidate_segmentation_debug_v1.tsv`
  - `analysis/table_authority_validation_v1.tsv`
  - `analysis/stage2_prompt_preview_v1.tsv`
  - `analysis/table_selection_debug_v1.json` when summary-table mode is used
- object families emitted by the authoritative Stage2 boundary:
  - `formulation_identity_candidate`
  - `component_candidate`
  - `phase_candidate`
  - `process_step_candidate`
  - `variable_or_factor_candidate`
  - `measurement_candidate`
  - `relation_cue`
  - `evidence_handoff`

Stage2 internal intermediate artifacts:

- `data/results/run_<run_id>/semantic_stage2_objects/candidate_blocks/<paper_key>/candidate_blocks_v1.json`
- `data/results/run_<run_id>/semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
- `data/results/run_<run_id>/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
- `data/results/run_<run_id>/semantic_stage2_objects/semantic_stage2_v2_summary.tsv`

S2-2 contract note:

- the maintained Stage2 path now formalizes S2-2:
  - clean text -> governed evidence package
- S2-2 is the first engineering freeze point in Stage2 and emits:
  - `semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`
  - `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
- the maintained Stage2 path now also formalizes an explicit internal boundary
  inside S2-2:
  - clean text / extracted tables -> candidate segmentation -> evidence-driven
    selector -> governed evidence package
- `candidate_blocks_v1.json` is the maintained pre-selector truth surface for
  candidate segmentation, table isolation, and conservative noise filtering
- `normalized_table_payloads_v1.json` is the maintained execution-facing
  full-table authority surface for formulation-relevant selected tables
- each preserved authority record must carry stable `table_id`,
  `source_table_reference`, deterministic `table_type`, `row_count`,
  `has_row_numbering`, `header_structure`, `raw_cells`, execution-facing
  `normalized_rows`, `row_identity_signals`, and `reconstruction_confidence`
- `evidence_blocks_v1.json` is the canonical pre-LLM truth surface for evidence
  selection, ordering, and packing
- table-derived evidence blocks must carry stable `table_id` and explicit
  `summary_is_lossy=true`
- candidate segmentation is responsible for candidate discovery plus
  execution-grade table preservation; selector semantics remain downstream
- S2-2a may apply conservative table-authority ranking over recovered table
  payloads so the preserved authority set marks stronger tables as primary and
  weaker but still distinct tables as secondary
- this ranking is structure and authority formation only:
  - it may use repair quality, row-anchor density, formulation-like row
    structure, and obvious downstream-result demotions
  - coarse labels such as `non_formulation_table` or
    `characterization/results` are noisy priors only and may demote but do not
    veto primary authority by themselves
  - only structural failure such as repair-insufficient payloads or narrative
    / figure spillover may block primary authority
  - it must not infer semantic roles, optimization meaning, or pre-LLM paper
    interpretation
- S2-2a owns the full-table authority surface and must preserve row numbering,
  row order, column structure, header hierarchy when available, and
  table-local identifiers when available
- S2-2b owns the semantic-facing summary or evidence view for selector and LLM
  packaging; it does not own lossless table preservation
- the maintained S2-2 selector is deterministic and evidence-driven:
  - conservative noise filtering
  - minimum evidence coverage
  - bounded packing
  - no required-role coverage contract
  - no archetype overlay in selection
  - no second LLM is used for this pre-LLM selection boundary
- the maintained selector may classify candidate tables only as:
  - `must_include`
  - `optional_context`
  - `hard_drop`
- irreversible table-preservation contract:
  - only confirmed pure noise may be hard-dropped
  - if a table is not confirmed noise, it must remain preserved in the
    pre-LLM authority surface
  - rules must not downrank, suppress, or remove a table because another table
    seems more important or more formulation-bearing
- bounded summary-view labels may still exist for observability, but they must
  not act as importance-based table vetoes
- candidate-level observability is maintained through
  `analysis/candidate_segmentation_debug_v1.tsv`
- selection is evidence-priority based rather than role-constrained, and
  semantically overlapping proxy or fallback blocks may be suppressed when
  authoritative evidence already exists
- `must_include` table summaries must be packed in neutral stable order, such
  as source or table-number order, and must not be semantically promoted to
  one true primary table before the LLM
- the maintained `S2-3` / `S2-4a` summary path is neutral across preserved
  tables; the main residual risk is lossy summary compression rather than
  cross-table importance bias
- summary blocks should preserve header / column schema and first-column row
  identity surfaces as the primary structure; sample rows are optional aids
- prompt assembly must consume the persisted S2-2 artifact rather than
  silently recomputing evidence from clean text
- this prompt-assembly boundary is S2-3:
  - it may assemble prompts from `evidence_blocks_v1.json` only
  - it must not reread clean text or perform new evidence selection or ranking
- semantic and execution split rule:
  - the LLM may see a lossy or compact summary view
  - downstream deterministic execution must resolve back to the S2-2
    full-table authority surface by stable table identity when row
    materialization is authorized
  - deterministic authority handles such as `authority_run_dir`,
    `authority_payload_root`, and table-scope locators belong to the
    execution-side contract and must not be treated as LLM semantic content
  - replay compatibility should reattach them through governed deterministic
    sidecars or reattachment surfaces, not by relying on the LLM to transmit
    them
- unified execution-input rule:
  - DOE and non-DOE row materialization must use the preserved S2-2
    full-table authority surface as the execution source of truth
  - Stage1 table assets may be reread only inside S2-2a reconstruction and
    validation; they are not the downstream execution input once authority
    exists
- engineering principle:
  the semantic-facing summary view is not the execution source of truth;
  deterministic execution operates on the preserved table entity
- current-cycle frozen-substep discoverability mapping:
  - `S2-2a`
    - owner:
      `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_candidate_segmentation_artifact`
    - outputs:
      `semantic_stage2_objects/candidate_blocks/<paper_key>/candidate_blocks_v1.json`
      `semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`
      `semantic_stage2_objects/normalized_table_payloads/<paper_key>/payloads/*.csv`
      `analysis/candidate_segmentation_debug_v1.tsv`
      and `analysis/table_authority_validation_v1.tsv`
    - stop boundary:
      candidate segmentation, conservative table-authority ranking, and
      execution-grade table preservation only
    - next lawful step:
      `S2-2b`
  - `S2-2b`
    - owner:
      `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_evidence_blocks_artifact`
      plus `build_evidence_priority_selection`
    - outputs:
      `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
      and `analysis/table_selection_debug_v1.json`
    - stop boundary:
      canonical semantic-facing evidence handoff written; full-table authority
      remains preserved from S2-2a
    - governed note:
      the maintained selector may enforce a minimal evidence sufficiency floor after evidence-priority ranking, but that floor remains evidence-only and must not emit semantic signals or semantic-role contracts before the LLM
    - next lawful step:
      `S2-3`
  - `S2-3`
    - owner:
      `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_live_prompt`
      plus `build_prompt_preview_row`
    - outputs:
      in-memory semantic-only prompt payload and maintained observability
      `analysis/stage2_prompt_preview_v1.tsv`
    - stop boundary:
      prompt assembled from canonical evidence only
      and runtime packing metadata recorded in audit surfaces rather than narrated to the LLM
    - next lawful step:
      `S2-4b live LLM call`, or explicit `S2-4a` prompt materialization when that frozen boundary is being audited
  - `S2-4a`
    - owner:
      `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py`
    - outputs:
      `analysis/s2_4a_prompt_template_v1.txt`
      `analysis/s2_4a_prompts_v1.jsonl`
      `analysis/s2_4a_prompt_audit_v1.tsv`
      and stage-local `RUN_CONTEXT.md`
    - stop boundary:
      prompt artifacts written, no live LLM call
    - governed note:
      all table evidence remains structural summary-only at this frozen
      boundary; generated summary metadata must not pre-label semantic table
      roles, and the prompt must explicitly assign semantic table scoping to the
      LLM rather than assuming deterministic pre-resolution
    - next lawful step:
      `S2-4b live LLM call`
  - `S2-4b`
    - owner:
      `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`
    - inputs:
      frozen `analysis/s2_4a_prompts_v1.jsonl`
    - outputs:
      replayable `raw_responses/<paper_key>__stage2_v2_raw_response.json`
      request metadata sidecars under `request_metadata/`
      `analysis/s2_4b_request_summary_v1.tsv`
      and stage-local `RUN_CONTEXT.md`
    - live-call policy for the current cycle:
      - model:
        explicit `--model` argument on the `S2-4b` live-call boundary; no
        repository-wide default model is applied
      - request mode:
        `stream_collect`
      - timeout seconds:
        `180`
      - retries:
        `0`
      - returned content is persisted without semantic judgment at this boundary
    - stop boundary:
      raw-response payloads written, no parsing or validation
    - next lawful step:
      `S2-5 semantic parsing`
  - `S2-5`
    - owner:
      `src/stage2_sampling_labels/run_stage2_s2_5_semantic_parsing_v1.py`
    - inputs:
      frozen `raw_responses/<paper_key>__stage2_v2_raw_response.json`
      plus minimal manifest provenance needed for paper metadata and `source_text_path`
    - outputs:
      `semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
      `semantic_stage2_objects/semantic_stage2_v2_summary.tsv`
      and stage-local `RUN_CONTEXT.md`
    - stop boundary:
      semantic-intermediate artifacts written, no contract validation or compatibility projection
    - next lawful step:
      `S2-6 contract validation`
  - `S2-6`
    - owner:
      `src/stage2_sampling_labels/run_stage2_s2_6_contract_validation_v1.py`
    - inputs:
      frozen `semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
      plus any preserved `S2-5` provenance sidecars already written in the same run directory
    - outputs:
      `analysis/stage2_semantic_authority_contract_report_v1.json`
      and stage-local `RUN_CONTEXT.md`
    - stop boundary:
      semantic contract validation artifacts written, no compatibility projection
    - next lawful step:
      `S2-7 compatibility projection`
  - `S2-7`
    - owner:
      `src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py`
    - inputs:
      passing `S2-6` validation report plus the referenced frozen
      `semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
    - outputs:
      `semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
      `semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.jsonl`
      `semantic_to_widerow_adapter/compatibility_projection_trace_v1.tsv`
      `semantic_to_widerow_adapter/compatibility_projection_summary_v1.json`
      and stage-local `RUN_CONTEXT.md`
    - stop boundary:
      completed Stage2 artifact written, no Stage3 execution
    - next lawful step:
      `Stage3 relation materialization`
- S2-4 is the LLM call boundary:
  - it is the only nondeterministic Stage2 substep
  - it emits run-scoped raw LLM response payloads
- S2-5 is semantic parsing of those raw responses into Stage2 semantic objects
- S2-6 is contract validation:
  - it is a legality and provenance gate, not selector logic
- S2-7 is compatibility projection:
  - it is the deterministic Stage3 handoff surface
  - it is not evidence construction
- `analysis/stage2_prompt_preview_v1.tsv` remains maintained observability, but
  it is derived from the same evidence artifact and is not the canonical source
- the S2-2 artifact records:
  - the resolved input contract
  - `selection_mode`
  - compact per-block evidence metadata
  - coverage summary
  - feature activation snapshot
  - `technical_status`
  - `design_status`
- downstream prompt assembly is normal only when the artifact is technically
  complete and the design-status evaluation is explicitly recorded
- segmentation closure freeze rule:
  once S2-2a segmentation closure is declared for the current cycle, candidate
  segmentation is frozen by default and follow-on closure work should target
  S2-2b selector/evidence prioritization unless a concrete segmentation
  regression is proven
- S2-2b strict stage-local debugging rule:
  - after S2-2a freeze, S2-2b proceeds as a stage-local loop:
    audit -> minimal selector-only fix -> rerun on the same frozen inputs
  - Audit is performed against frozen S2-2 artifacts and a fixed human reference passage set (`docs/selector_calibration/`), independent of downstream validation.
  - selector works only on existing `candidate_blocks_v1.json`; it must not
    introduce new candidate discovery behavior
  - the audit uses stage-local S2-2 artifacts only
  - closure means stage-local selector freeze only
- downstream validation note:
  live Stage2 runs that traverse S2-3 through S2-7 are separate later tasks
  and must not be used for S2-2b debugging or closure judgment

Boundary status:

- `semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
  - `internal_intermediate`
- `semantic_stage2_objects/semantic_stage2_v2_summary.tsv`
  - `internal_intermediate`
- `semantic_stage2_objects/raw_responses/`
  - `diagnostic_boundary`
  - replayable only through the maintained composite Stage2 replay path
  - does not become a lawful Stage3 upstream boundary until that replay path
    re-emits the completed Stage2 artifact

Stage completion artifact:

- completed Stage2 candidate surface:
  - `data/results/run_<run_id>/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`

Boundary status:

- `weak_labels__v7pilot_r3_fixparse.tsv`
  - `mainline_resume_boundary`

Consumed by downstream stage:

- Stage 3

Legacy note:

- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
  is deprecated and not part of the active pipeline authority
- it may be used only for fallback or debug scenarios that are explicitly
  declared as non-mainline

### Stage 3. Deterministic Formulation Relation Materialization

Purpose:
Materialize explicit paper-level formulation relation structure from Stage 2
candidate rows without any LLM usage. This stage exists to separate relation
reasoning from later Stage 5 final-row closure and export.

Stage3 relation-binding extension:

- consume deterministic DOE rows and deterministic non-DOE table rows from the
  same completed Stage2 boundary
- propagate selected-condition inheritance from parent tables into child-table
  rows when authorized by Stage2 markers
- attach preparation-context shared fields only as deterministic inheritance,
  never as cross-table Cartesian reconstruction

Exact input files or directories:

- compatibility-projected Stage 2 legacy wide-row TSV
- optional compatibility-projected Stage 2 legacy wide-row JSONL
- optional scope manifest TSV for paper title and source-path enrichment

Exact script path(s) and script filename(s):

- `src/stage3_relation/build_formulation_relation_artifacts_v1.py`
- optional reproducibility wrapper:
  - `src/stage3_relation/run_formulation_relation_artifacts_v1.py`

Exact output files or directories:

- `formulation_relation_records_v1.tsv`
- `formulation_logic_graph_v1.jsonl`
- `formulation_relation_summary_v1.tsv`
- `resolved_relation_fields_v1.tsv`

Stage completion artifact:

- the Stage 3 relation artifact set:
  - `formulation_relation_records_v1.tsv`
  - `resolved_relation_fields_v1.tsv`

Consumed by downstream stage:

- Stage 5 as required deterministic relation input and required resolved-field materialization input
- human audit and failure localization

### Stage 4. Candidate-Level Diagnostics And Review Surfaces

Purpose:
Evaluate candidate-instance behavior and produce reviewer-facing diagnostic
artifacts. This stage is diagnostic and review-oriented. It is not the
benchmark-valid endpoint.

Exact input files or directories:

- compatibility-projected Stage 2 legacy wide-row TSV
- fixed manual GT workbook for the declared scope
- scope manifest TSV

Exact script path(s) and script filename(s):

- `src/stage4_eval/eval_weak_labels_v7pilot3.py`
- `src/stage4_eval/build_dev15_review_workbook_v1.py`

Exact output files or directories:

- per-run or per-scope diagnostic outputs such as:
  - `per_doi_formulation_instance_summary.tsv`
  - `predicted_instance_rows.tsv`
  - reviewer workbooks under `data/results/dev15_review/`

Stage completion artifact:

- candidate diagnostic summary TSV for the declared scope

Consumed by downstream stage:

- human review and debugging only

### Stage 5. Final Formulation Closure And Benchmark Comparison

Purpose:
Convert candidate formulation-instance rows into final one-row-per-formulation
records and auditable value-layer sidecars without changing the formulation
universe authorized upstream.

Stage 5 is a materialization, validation, and final-table closure layer. It must
not perform semantic inference for formulation membership or same-paper donor
search. Optional LLM use inside Stage5 is allowed only after the row universe is
fixed and only as direct value candidate extraction with evidence, scope, and
validator review.

Benchmark-validity rule:

- Stage5 final-table generation is necessary but not sufficient for
  benchmark-valid reporting.
- In the current diagnostic-development phase, the maintained GT compare node
  consumes only the completed Stage5 final table, declared scope manifest, and
  frozen GT authority.
- Benchmark-valid status remains disabled until a governed benchmark contract is
  explicitly re-enabled.

Internal Stage5 family rule:

- benchmark-final family:
  - `build_minimal_final_output_v1.py`
  - `compare_final_table_to_gt_v1.py`
  - `build_minimal_final_output_v1.py` now emits two governed sibling outputs:
    - primary benchmark-facing `final_formulation_table_v1.tsv`
    - linked lower-level `downstream_variant_records_v1.tsv` for excluded downstream/post-processing descendants
- downstream modeling-ready family:
  - first maintained modeling-ready surface: `src/stage5_benchmark/build_modeling_ready_sidecar_v1.py`
  - this helper reads only the frozen benchmark-final table and emits a downstream sidecar of deterministic parse/math transforms with row identity linkage and raw-value provenance
  - first maintained row-wise modeling-ready table: `src/stage5_benchmark/build_modeling_ready_table_v1.py`
  - this helper reads the frozen benchmark-final table plus the sidecar and emits one row per frozen formulation with selected raw carry-through values and selected transformed modeling columns
  - deterministic normalization, derivation, and curated projection helpers
    that operate only downstream of the frozen benchmark-final object
- downstream audit/review family:
  - audit-ready export and reviewer workbooks derived from the frozen
    benchmark-final object
- these are internal Stage5 families only; they do not create a new coarse
  stage or alter benchmark comparison semantics

Stage 5 identity constraints layer:

- exclude parent-linked non-synthesis descendants when they are explicitly
  downstream/control/characterization references to an existing parent
  formulation identity
- preserve those excluded downstream/post-processing descendants in the linked
  lower-level `downstream_variant_records_v1.tsv` surface instead of silently
  dropping them
- exclude unparented shared-condition summaries and comparative summary
  references when they do not define an independently reported formulation
  instance

Final-row attachment discipline:

- once the Stage5 final-row universe is closed, downstream stages must attach values, not reconstruct
  formulations
- formulation count and membership must remain invariant after Stage5 final-table closure
- downstream stages may add fields, resolve missing fields, and derive fields
- downstream stages must not:
  - split formulations implicitly
  - merge formulations implicitly
  - create new formulations from value similarity
- only explicitly authorized identity-defining fields may justify a split
- measurement fields such as size, PDI, zeta, EE, and LC must not trigger a
  split by default
- if uncertain, attach to the existing identity rather than split

Benchmark-final contract:

- the canonical benchmark-valid object is `final_formulation_table_v1.tsv`
- benchmark-final may include:
  - deterministic row closure
  - identity-preserving filtering
  - conservative duplicate or variant collapse under explicit rules
  - explicit Stage3 resolved relation carry-through
- benchmark-final must not:
  - silently replace paper-reported values with convenience-normalized values
  - perform assumption-based inference
  - perform donor-fill
  - change formulation membership after Stage5 final-table closure

Downstream modeling-ready contract:

- modeling-ready outputs must be downstream of the frozen benchmark-final
  object
- the first maintained modeling-ready path is a sidecar TSV built from
  `final_formulation_table_v1.tsv` plus explicit deterministic transform rules
- the first maintained row-wise modeling-ready table is built from that frozen
  final table plus the sidecar and keeps raw benchmark-final values distinct
  from transformed modeling columns
- they may include normalization, canonical label cleanup, unit harmonization,
  safe deterministic derivation, and curated projection
- they must preserve raw benchmark-final values and provenance
- they must not replace `final_formulation_table_v1.tsv`
- they must not change formulation membership

Stage 5 post-comparison risk stratification layer:

- build paper-level Layer 2 audit-risk labels from an existing Layer 2
  identity-comparison artifact
- do not change Stage 2 extraction, Stage 3 relation materialization, Stage 5
  final-table closure, or benchmark-valid comparison counts
- use the resulting risk labels only as downstream audit-use metadata for
  Layer 3 field-level GT work

Exact input files or directories:

- compatibility-projected Stage 2 legacy wide-row TSV
- required Stage 3 relation-record TSV
- required Stage 3 resolved relation-field TSV
- scope manifest TSV for the declared benchmark scope

Exact script path(s) and script filename(s):

- `src/stage5_benchmark/build_minimal_final_output_v1.py`
- optional `NON-CANONICAL, STAGE5_ONLY` convenience helper:
  - `src/stage5_benchmark/run_minimal_final_output_v1.py`
- optional post-comparison risk helper:
  - `src/stage5_benchmark/build_layer2_risk_assessment_v1.py`
- optional Layer 2 GT review export helper:
  - `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`

Exact output files or directories:

- `final_formulation_table_v1.tsv`
- `final_output_decision_trace_v1.tsv`
- `final_output_summary_v1.md`
- optional comparison-metadata outputs:
  - `analysis/paper_risk_assessment.tsv`
  - `analysis/paper_risk_assessment_summary.md`
- optional downstream Layer 3 review-pack outputs built from the frozen final
  table plus comparison metadata:
  - `final_formulation_table_audit_ready_v1.tsv`
  - `field_gt_review_seed_rows_v*.tsv`
  - `field_gt_review_source_summary_v*.tsv`
  - `field_gt_review_workbook_v*.xlsx`

Reviewer-facing audit-system interpretation:

- these Layer 3 artifacts are not only evaluation aids
- together they form part of the governed post-comparison audit and governance
  layer around the frozen formulation database
- the benchmark-valid production-path endpoint remains
  `final_formulation_table_v1.tsv`
- reviewer-facing audit outputs must remain downstream of the frozen final
  table and must not mutate benchmark-valid outputs
- current design is formulation-centered in intent, but still fragmented in
  implementation across:
  - paper-level risk
  - formulation-level audit-ready export
  - field-level review workbook
  - cell-level cross-audit report
  - evidence handoff tooling
- these downstream reviewer-facing surfaces are not modeling-ready projections
  and are not benchmark-final builders; they are frozen-final-table audit
  surfaces

Stage completion artifact:

- final formulation table:
  - `final_formulation_table_v1.tsv`

Consumed by downstream stage:

- the GT comparison node
- downstream audit/review surfaces that explicitly consume the frozen Stage5 final table

Post-Stage5 diagnostic layer:

- `src/stage5_benchmark/compare_final_table_to_gt_v1.py`
- required inputs:
  - `final_formulation_table_v1.tsv`
  - frozen Layer1 GT counts TSV
  - declared scope manifest TSV
- maintained compare-mode clarification:
  - diagnostic mode is the active compare mode in the current phase
  - benchmark mode is disabled until a governed benchmark contract is explicitly re-enabled

## Evaluation Reference Path

The evaluation reference path is not a production stage. It is the fixed manual
benchmark input surface used only for comparison and diagnosis.

Reference inputs:

- `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v2_variantaware.xlsx`
- other checked manual assets under `data/cleaned/labels/manual/` when the
  declared scope requires them
- declared scope manifest TSV

Runtime rule:

- these assets are reference inputs, not system-produced transformation outputs
- they must not be described as a canonical system stage

## Comparison Node

Purpose:
Compare the Stage 5 final formulation table against the frozen Layer1 GT counts
TSV for the declared scope.

Exact script path and filename:

- `src/stage5_benchmark/compare_final_table_to_gt_v1.py`

Comparison inputs:

- `final_formulation_table_v1.tsv`
- frozen Layer1 GT counts TSV
- declared scope manifest TSV

Comparison outputs:

- `final_table_vs_gt_counts.tsv`
- `final_table_vs_gt_summary.md`

Compare contract:

- diagnostic mode is the active compare mode in the current phase
- diagnostic compare continuation must never be reported as benchmark-valid

Optional post-compare review surface:

- `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`
  - consumes the Stage 5 final formulation table, optional decision trace, and optional relation records
  - writes a run-scoped XLSX boundary-GT review workbook for manual Layer 2 curation
  - is reviewer-facing and diagnostic, not a benchmark-valid endpoint by itself

## Full Chain From Raw Zotero Records To Final Outputs

The production chain is:

1. raw Zotero-derived records under `data/raw/zotero/`
2. authoritative manifest and cleaned text assets under `data/cleaned/`
3. Stage 2 semantic-intermediate artifacts under `data/results/run_<run_id>/`
4. Stage 2 completed downstream-ready artifacts under
   `data/results/run_<run_id>/`
5. Stage 3 relation-record artifact set
6. optional Stage 4 diagnostic artifacts under `data/results/`
7. Stage 5 final formulation table under `data/results/run_<run_id>/`

The final deliverable of the production path is:

- the final formulation-level structured dataset:
  - `final_formulation_table_v1.tsv`

The separate comparison node then consumes:

- `final_formulation_table_v1.tsv`
- fixed manual GT workbook
- declared scope manifest TSV

and produces:

- `final_table_vs_gt_counts.tsv`
- `final_table_vs_gt_summary.md`

## Manual Reproduction

The pipeline is manually reproducible stage by stage. No hidden orchestration
script is part of the canonical contract.

### Step 0. Build or refresh the raw Zotero-derived corpus record set

```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'
python src/stage0_relevance/zotero_api_sync_selected.py
python src/stage0_relevance/zotero_fetch_llm_relevant_pdfs.py
```

If a raw Zotero-derived JSONL already exists and is declared unchanged, it may
be reused instead of re-running Stage 0.

### Step 1. Build the authoritative manifest and cleaned text assets

```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'
python src/stage1_cleaning/zotero_raw_to_manifest.py --overwrite
python src/stage1_cleaning/clean_manifest_to_text.py --overwrite --prefer html
python src/stage1_cleaning/run_tables_extraction_for_dataset_v1.py --dataset-id goren_2025 --manifest-tsv data/cleaned/goren_2025/index/manifest.tsv
python src/stage1_cleaning/hydrate_manifest_v1.py --manifest-tsv data/cleaned/index/manifest_current.tsv --out-tsv data/cleaned/index/manifest_current.tsv --overwrite --dataset-manifest-tsv data/cleaned/goren_2025/index/manifest.tsv --dataset-id goren_2025 --split-manifest-tsv data/cleaned/goren_2025/index/splits/dev_manifest_v7pilot3_2026-03-06.tsv --split-tag dev_manifest_v7pilot3_2026-03-06 --benchmark-tag DEV15 --split-manifest-tsv data/cleaned/goren_2025/index/splits/dev_manifest_remaining12_2026-03-10.tsv --split-tag dev_manifest_remaining12_2026-03-10 --benchmark-tag DEV15
```

When the canonical manifest is assembled from multiple declared upstream raw
sources, pass repeatable `--input` arguments plus aligned source-provenance
arguments such as `--source-collection`, `--source-manifest-lineage`,
`--source-selection-rule`, and `--input-dataset-id`. The assembly contract is
explicit; it must not be inferred from recency, file naming similarity, or
undocumented notes.

Manifest hydration rule:

- `S1-2` assembly alone is not the contract-complete canonical manifest.
- `S1-3a` asset hydration must bind rows to governed cleaned text and table
  surfaces such as `data/cleaned/index/key2txt.tsv` and dataset-scoped table
  roots.
- `S1-3b` scope overlays must bind deterministic dataset/split/benchmark
  metadata from governed dataset and split manifests.

### Step 2. Run the composite Stage2 entrypoint

Use a declared manifest or split manifest for the chosen scope.

```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'
python src/stage2_sampling_labels/run_stage2_composite_v1.py --manifest-tsv <scope_manifest.tsv> --paper-key <paper_key_1> --paper-key <paper_key_2> --source-mode <live_llm_or_legacy_llm_replay> --llm-backend <gemini_or_nvidia> --model <model_name> --max-text-chars <max_chars>
```

Default writer rule:

- when no explicit output path is supplied, this maintained entrypoint now
  allocates a future-facing MDEC084 child execution path under
  `data/results/YYYYMMDD_<short_hash>/NN_stage2/`
- explicit `--run-id run_...` remains legacy compatibility mode only
- explicit `--run-dir <data/results/...>` remains allowed

Authority rule:

- The authoritative Stage2 contract is composite:
  - internal LLM semantic discovery
  - internal deterministic post-LLM completion
- Stage2 evaluation for downstream readiness must target the completed Stage2
  artifact after deterministic completion.
- Direct comparison of raw LLM semantic objects to formulation-level GT is
  diagnostic-only failure localization, not authoritative completed-Stage2
  evaluation.

### Step 3. Build the Stage 3 deterministic relation artifacts

The Stage 3 layer is deterministic and must not call any LLM or external API.

```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'
python src/stage3_relation/run_formulation_relation_artifacts_v1.py --run-id <stage3_run_id> --out-subdir formulation_relation_v1 --weak-labels-tsv data/results/<stage2_run_id>/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv --weak-labels-jsonl data/results/<stage2_run_id>/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.jsonl --scope-manifest-tsv <scope_manifest.tsv>
```

### Step 4. Generate candidate-level diagnostics if needed

This step is optional for production-path completion but is part of the
canonical debug surface.

```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'
python src/stage4_eval/eval_weak_labels_v7pilot3.py --pilot-tsv data/results/<stage2_run_id>/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv --pilot-manifest <scope_manifest.tsv> --out-dir data/results/<stage4_run_id>
```

### Step 5A. Build the final formulation table

```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'
python -m src.stage5_benchmark.build_minimal_final_output_v1 --input-tsv data/results/<stage2_run_id>/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv --relation-records-tsv data/results/<stage3_run_id>/formulation_relation_v1/formulation_relation_records_v1.tsv --resolved-relation-fields-tsv data/results/<stage3_run_id>/formulation_relation_v1/resolved_relation_fields_v1.tsv --out-dir data/results/<final_run_id>
```

Stage 5 must fail fast if either Stage 3 relation artifact is missing. Silent
bypass of the relation layer is not allowed.

### Step 5B. Run the comparison node against GT

```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'
python src/stage5_benchmark/compare_final_table_to_gt_v1.py --final-table-tsv data/results/<final_run_id>/final_formulation_table_v1.tsv --gt-counts-tsv data/cleaned/gt_authority/v1/dev15_layer1_gt_counts.tsv --scope-manifest-tsv <scope_manifest.tsv> --out-dir data/results/<final_run_id> --scope-name <declared_scope_name>
```

The production path ends at Step 5A.
Benchmark-valid reporting requires:

- Step 5A final-table materialization
- the separate Step 5B comparison node
- explicit `benchmark` compare mode or another later mode that preserves the
  same legality rule

## Provenance And Valid Incremental Reuse

- Final evaluation must still follow the canonical complete pipeline path.
- Unchanged upstream artifacts may be reused.
- Every reused artifact must be explicitly declared in the run context.
- Reused artifacts must remain traceable to their producing script and run.
- Caching is allowed.
- Bypassing required stages is not allowed.

## Run-Lineage Containment

- One top-level `data/results/run_*` directory represents one declared lineage.
- Retries, repair passes, partial reruns, deterministic refreshes, and stage-only
  child executions for that same lineage must stay under the parent lineage
  directory rather than creating more sibling top-level run directories.
- Recommended child layout:
  - `data/results/<parent_run_id>/lineage/children/<ordered_role>/<child_run_id>/`
- The parent lineage directory must remain the authoritative entrypoint for
  human inspection and must expose a child-step index or mapping when nested
  child runs exist.

Examples of valid reuse:

- reuse a checked raw Zotero-derived JSONL instead of re-running Stage 0
- reuse a checked manifest and cleaned text set instead of re-running Stage 1
- reuse an unchanged Stage 2 candidate TSV while iterating on Stage 3 or Stage 5
- reuse the fixed manual GT workbook as a declared reference input

Examples of invalid reuse:

- using an undocumented ad hoc TSV as the canonical Stage 2 completion artifact
- skipping Stage 5B and treating Stage 5A output as the reported benchmark result
- mixing a Stage 2 TSV from one scope with a GT workbook from another scope

## Forbidden Shortcuts

- Do not compare Stage 2 candidate extraction rows directly to GT as if they
  were final system outputs.
- Do not skip required pre-evaluation stages and report partial artifacts as the
  benchmark result.
- Do not use undocumented ad hoc files as canonical stage outputs.
- Do not hide the canonical pipeline inside a single end-to-end Python wrapper.
- Do not treat forced full recomputation as the definition of a complete
  pipeline. Completeness is provenance and stage coverage, not re-running every
  stage every time.

## Current Controlled Benchmark Inputs

For the current DEV-oriented engineering work, the most important checked inputs
are:

- `data/raw/zotero/zotero_selected_items.jsonl`
- `data/cleaned/index/manifest_current.tsv`
- `data/cleaned/goren_2025/index/splits/dev_manifest_v7pilot3_2026-03-06.tsv`
- `data/cleaned/goren_2025/index/splits/dev_manifest_remaining12_2026-03-10.tsv`
- reference GT workbook:
  - `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v2_variantaware.xlsx`

## Stage 3 Scope Boundary

The Stage 3 relation layer is intentionally lightweight in phase 1.

- It materializes explicit method-group, parent-link, shared-field, and
  variation-axis relations from Stage 2 weak labels.
- It does not call any LLM or perform final benchmark comparison.
- It improves auditability and provides optional provenance to Stage 5.
- The current Stage 5 closure logic does not yet fully drive keep/drop/collapse
  decisions from the relation artifact itself; that remains a future extension.

## Layer 1 GT Counting Rule

Layer 1 GT counts are based on reported formulation instances, not on the full
experimental design space.

- Include a GT formulation only when the paper reports it as an experimental
  instance, such as a table row or an explicit condition tied to results.
- Exclude conditions that are described only as possible combinations, sweep
  coordinates, or methods-level design space without instance-level evidence.
