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
   Stage 3 relation provenance

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
- input B: fixed manual GT workbook
- input C: declared scope manifest TSV
- output: `final_table_vs_gt_counts.tsv` and `final_table_vs_gt_summary.md`
- optional downstream comparison metadata:
  - `analysis/paper_risk_assessment.tsv`
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

- `data/raw/zotero/zotero_selected_items.jsonl`
- local PDF and HTML files referenced by the raw records

Exact script path(s) and script filename(s):

- `src/stage1_cleaning/zotero_raw_to_manifest.py`
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
Stage2 contains both:

1. LLM semantic discovery
2. deterministic post-LLM completion into the downstream-ready Stage2 artifact

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

Exact input files or directories:

- cleaned manifest or split manifest TSVs
- cleaned text assets
- cleaned table assets when available

Exact script path(s) and script filename(s):

- `src/stage2_sampling_labels/run_stage2_composite_v1.py`

Internal Stage2 scripts:

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
- canonical semantic-object artifacts:
  - `semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
  - `semantic_stage2_objects/semantic_stage2_v2_summary.tsv`
  - `semantic_stage2_objects/raw_responses/`
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

- `data/results/run_<run_id>/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
- `data/results/run_<run_id>/semantic_stage2_objects/semantic_stage2_v2_summary.tsv`

Boundary status:

- `semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
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
records.

Stage 5 is a materialization layer. It must not perform semantic inference or
same-paper donor search.

Stage 5 identity constraints layer:

- exclude parent-linked non-synthesis descendants when they are explicitly
  downstream/control/characterization references to an existing parent
  formulation identity
- exclude unparented shared-condition summaries and comparative summary
  references when they do not define an independently reported formulation
  instance

Identity freeze and attachment discipline:

- once formulation identity is frozen by the reviewed Layer2-style boundary
  authority, downstream stages must attach values, not reconstruct
  formulations
- formulation count and membership must remain invariant after identity freeze
- downstream stages may add fields, resolve missing fields, and derive fields
- downstream stages must not:
  - split formulations implicitly
  - merge formulations implicitly
  - create new formulations from value similarity
- only explicitly authorized identity-defining fields may justify a split
- measurement fields such as size, PDI, zeta, EE, and LC must not trigger a
  split by default
- if uncertain, attach to the existing identity rather than split

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
- optional identity-freeze guardrail helper:
  - `src/stage5_benchmark/enforce_identity_freeze_v1.py`

Exact output files or directories:

- `final_formulation_table_v1.tsv`
- `final_output_decision_trace_v1.tsv`
- `final_output_summary_v1.md`
- optional comparison-metadata outputs:
  - `analysis/paper_risk_assessment.tsv`
  - `analysis/paper_risk_assessment_summary.md`
- optional identity-freeze guardrail outputs:
  - `identity_freeze_report_v1.tsv`
  - `identity_freeze_summary_v1.tsv`
  - `identity_freeze_summary_v1.md`
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

Stage completion artifact:

- final formulation table:
  - `final_formulation_table_v1.tsv`

Consumed by downstream stage:

- the mandatory identity freeze gate

Mandatory post-Stage5 gate:

- `src/stage5_benchmark/enforce_identity_freeze_v1.py`
- required inputs:
  - upstream identity scaffold surface
  - `final_formulation_table_v1.tsv`
- required behavior:
  - fail the run on any:
    - row count drift
    - identity reassignment
    - unresolved scaffold binding
    - ambiguous scaffold binding
- if the gate fails, the run is invalid for downstream value-level evaluation
  and reviewer-facing export
- only after the gate passes may the pipeline continue to:
  - comparison
  - audit-ready export
  - Layer 3 review/evaluation surfaces

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
Compare the Stage 5 final formulation table against the fixed manual GT workbook
for the declared scope.

Exact script path and filename:

- `src/stage5_benchmark/compare_final_table_to_gt_v1.py`

Comparison inputs:

- `final_formulation_table_v1.tsv`
- fixed manual GT workbook
- declared scope manifest TSV

Comparison outputs:

- `final_table_vs_gt_counts.tsv`
- `final_table_vs_gt_summary.md`

Optional post-compare review surface:

- `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`
  - consumes the Stage 5 final formulation table, optional decision trace, and optional relation records
  - writes a run-scoped XLSX boundary-GT review workbook for manual Layer 2 curation
  - is reviewer-facing and diagnostic, not a benchmark-valid endpoint by itself
- `src/stage5_benchmark/enforce_identity_freeze_v1.py`
  - consumes an upstream identity scaffold surface plus the Stage 5 final
    formulation table
  - emits row-count drift, identity-reassignment, and violation diagnostics
  - does not silently fix or mutate benchmark-valid outputs

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
```

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
python src/stage5_benchmark/compare_final_table_to_gt_v1.py --final-table-tsv data/results/<final_run_id>/final_formulation_table_v1.tsv --gt-xlsx data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v2_variantaware.xlsx --scope-manifest-tsv <scope_manifest.tsv> --out-dir data/results/<final_run_id> --scope-name <declared_scope_name>
```

The production path ends at Step 5A.
Benchmark-valid reporting requires the separate Step 5B comparison node.

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
