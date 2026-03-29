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
- Stage 2 semantic-object generation
- deterministic compatibility projection into the legacy wide-row surface
- Stage 3 deterministic formulation relation materialization layer
- Stage 4 candidate-instance diagnostics and review surfaces
- Stage 5 final formulation-table closure
- a separate Stage 5 comparison node that reads the final formulation table and
  fixed manual GT workbook as separate inputs

The canonical benchmark object is the Stage 5 final formulation table. No
intermediate artifact may be reported as the system result against GT.

## Canonical Path Definition

The system canonical production path is:

1. Stage 0: build or refresh Zotero-derived raw records and local source assets
2. Stage 1: convert raw records into the authoritative manifest, cleaned text,
   and table assets
3. Stage 2: emit semantic formulation objects from the cleaned corpus
4. Compatibility bridge: deterministically project semantic Stage2 objects into
   the legacy wide-row surface required by unchanged downstream consumers
5. Stage 3: materialize explicit paper-level formulation relation artifacts
   from the compatibility-projected legacy wide-row surface
6. Stage 4: optionally generate candidate-level diagnostic and reviewer-facing
   artifacts
7. Stage 5: close candidate rows into final formulation rows with optional
   Stage 3 relation provenance

Active-transition note:

- Stage2.5 is retired from the active mainline and remains archived only as a
  historical exploratory side-path.
- The semantic Stage2 emitter is the authoritative Stage2 boundary.
- The deterministic compatibility adapter is part of the active mainline
  execution chain but is not itself a numbered semantic stage.
- The legacy wide-row Stage2 extractor is deprecated and retained only for
  fallback or debug use.

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

### Stage 2. Semantic Formulation Object Generation

Purpose:
Emit paper-driven semantic formulation objects from cleaned paper content and
tables. Stage2 is the authoritative semantic discovery layer and no longer
uses the legacy wide-row TSV as its primary output contract.

Exact input files or directories:

- cleaned manifest or split manifest TSVs
- cleaned text assets
- cleaned table assets when available

Exact script path(s) and script filename(s):

- `src/stage2_sampling_labels/sample_from_manifest_html_first.py`
- `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`

Exact output files or directories:

- run-scoped semantic-object outputs under
  `data/results/run_<run_id>/semantic_stage2_objects/`
- canonical semantic-object artifacts:
  - `semantic_stage2_objects/semantic_stage2_objects_v1.jsonl`
  - `semantic_stage2_objects/semantic_stage2_object_summary_v1.tsv`
  - `semantic_stage2_objects/semantic_stage2_object_manifest_v1.json`
- object families emitted by the authoritative Stage2 boundary:
  - `formulation_identity_candidate`
  - `component_candidate`
  - `phase_candidate`
  - `process_step_candidate`
  - `variable_or_factor_candidate`
  - `measurement_candidate`
  - `relation_cue`
  - `evidence_handoff`

Stage completion artifact:

- run-scoped semantic-object JSONL:
  - `data/results/run_<run_id>/semantic_stage2_objects/semantic_stage2_objects_v1.jsonl`

Consumed by downstream stage:

- deterministic compatibility bridge

### Step 2.5. Compatibility Bridge (Not A Semantic Stage)

Purpose:
Deterministically project semantic Stage2 objects into the legacy wide-row
surface required by unchanged Stage3, Stage4 diagnostic, and Stage5 runtime
consumers during migration.

Exact input files or directories:

- Stage2 semantic-object JSONL
- optional semantic summary and manifest sidecars

Exact script path(s) and script filename(s):

- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`

Exact output files or directories:

- compatibility-projected legacy wide-row artifacts under
  `data/results/run_<run_id>/semantic_to_widerow_adapter/`
- canonical bridge outputs:
  - `semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
  - `semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.jsonl`
  - `semantic_to_widerow_adapter/compatibility_projection_trace_v1.tsv`
  - `semantic_to_widerow_adapter/compatibility_projection_summary_v1.json`

Bridge boundary rules:

- the adapter must remain deterministic
- the adapter must not perform semantic inference
- the adapter must not invent evidence ownership or normalization
- the adapter exists to preserve downstream reuse while Stage3 and Stage5
  remain unchanged

Legacy note:

- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
  is deprecated and not part of the active pipeline
- it may be used only for fallback or debug scenarios that are explicitly
  declared as non-mainline

### Stage 3. Deterministic Formulation Relation Materialization

Purpose:
Materialize explicit paper-level formulation relation structure from Stage 2
candidate rows without any LLM usage. This stage exists to separate relation
reasoning from later Stage 5 final-row closure and export.

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

Stage completion artifact:

- final formulation table:
  - `final_formulation_table_v1.tsv`

Consumed by downstream stage:

- the Stage 5 comparison node

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

## Full Chain From Raw Zotero Records To Final Outputs

The production chain is:

1. raw Zotero-derived records under `data/raw/zotero/`
2. authoritative manifest and cleaned text assets under `data/cleaned/`
3. Stage 2 semantic-object artifacts under `data/results/run_<run_id>/`
4. compatibility-projected legacy wide-row artifacts under
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

### Step 2. Produce semantic Stage2 objects

Use a declared manifest or split manifest for the chosen scope.

```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'
python src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py --manifest-tsv <scope_manifest.tsv> --out-dir data/results/<stage2_run_id>/semantic_stage2_objects --verbose
```

The authoritative Stage2 outputs are semantic objects, not a wide-row TSV.
Stage2 should preserve paper-reported structure and raw expressions rather than
forcing legacy projection decisions.

### Step 2.5. Run the deterministic compatibility bridge

```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'
python src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py --semantic-jsonl data/results/<stage2_run_id>/semantic_stage2_objects/semantic_stage2_objects_v1.jsonl --out-dir data/results/<stage2_run_id>/semantic_to_widerow_adapter
```

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
