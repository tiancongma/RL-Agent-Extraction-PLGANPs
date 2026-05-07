# PLGA End-to-End Boundary Repair Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task. Do not run live LLM calls without explicit batch-level user approval.

**Goal:** Re-establish an explicit, auditable end-to-end PLGA extraction flow from original source assets to `final_formulation_table_v1.tsv`, then repair the currently exposed Stage1/Stage2 boundary failures before returning to Stage5 value closure.

**Architecture:** This plan preserves the governed architecture: Stage1/Stage2 own source visibility and evidence authority; LLMs provide semantic signals rather than concrete numeric transmission; deterministic sidecars reattach LLM signals to full S2-2 table/value authority; Stage3 resolves relations and inheritance; Stage5 closes values on a fixed row universe. Audit findings are separated from the step definitions so the pipeline contract remains readable.

**Tech Stack:** Python standard library, repository-maintained `src/stage*_*/` scripts, TSV/JSON/JSONL artifacts, `python3 -m unittest`, run-scoped diagnostic outputs under `data/results/<run_id>/`.

---

## 0. Hard Constraints

- Do not create new governance files under `project/`.
- Do not infer authority by newest directory, lexical sort, mtime, glob-first match, or human memory.
- Use explicit CLI paths or `data/results/ACTIVE_RUN.json` for any workflow that consumes `data/results/` artifacts.
- All work in this plan is diagnostic/development until a governed full lineage is explicitly selected.
- No benchmark-valid claim may be made from intermediate Stage1/Stage2/Stage3 artifacts.
- No paper-key-specific runtime patching. DEV15 papers and user excerpts are validation anchors only.
- Clean text and Stage2 authority surfaces remain the source for downstream extraction; do not bypass them with Stage5 raw-source mining.
- Prompt summaries may be lossy; full numeric/table authority belongs to S2-2 payload/grid artifacts, not to the prompt.
- S5-3 live LLM value extraction is not run in this plan.

---

# Part A — Ordered End-to-End Pipeline Contract

This section defines the expected whole flow in order. Audit findings are intentionally not interleaved here.

## S0-1 Raw source intake

**Input**

- Zotero records from one or more Zotero libraries / collections / folders.
- PDF attachments.
- HTML attachments.
- Any explicitly declared raw source collection.

**Output**

- Checked raw Zotero-derived records such as:
  - `data/raw/zotero/zotero_selected_items.jsonl`
  - `data/raw/zotero/zotero_llm_relevant.jsonl`
- Attachment paths recorded in the raw records.

**Function**

- Build the raw item universe and local attachment availability surface.
- Preserve bibliographic identity and attachment lineage.

**Principles**

- Multiple Zotero libraries/folders may feed the corpus.
- This step does not define benchmark scope by itself.
- Raw intake is reproducibility input, not a downstream authority shortcut.

---

## S1-1 Canonical manifest assembly

**Input**

- One or more declared raw Zotero JSONL inputs.
- Declared source collection / folder / library metadata.
- Declared selection rules and dataset IDs.

**Output**

- One canonical authoritative TSV manifest:
  - `data/cleaned/index/manifest_current.tsv`

**Function**

- Merge multiple Zotero libraries/folders into one canonical paper universe.
- Assign stable `paper_key`, DOI/title identity, dataset/source lineage, and attachment references.

**Principles**

- There should be one authoritative manifest TSV, not multiple competing manifests.
- Subsets are selected later by explicit parameters, tags, split columns, benchmark tags, or run-scoped scope manifests derived from this canonical manifest.
- The canonical manifest must be explicit about source provenance for each row.

---

## S1-2 Clean text construction

**Input**

- `manifest_current.tsv`.
- PDF/HTML attachments referenced by manifest rows.

**Output**

- Cleaned text files under:
  - `data/cleaned/content/text/`
- Text mapping:
  - `data/cleaned/index/key2txt.tsv`

**Function**

- Convert PDF/HTML source into stable cleaned text.
- Preserve method sections, materials/preparation paragraphs, table captions, table-near text, results paragraphs, and source lineage.

**Principles**

- Remove confirmed format noise without deleting source evidence.
- Preserve section structure and table-local context when possible.
- Record whether text came from PDF, HTML, or another declared source.
- If important information is absent from clean text, the repair belongs upstream, not in Stage5.

---

## S1-3 Table asset extraction

**Input**

- Source PDF/HTML assets.
- Canonical manifest rows.
- Dataset-local extraction settings.

**Output**

- Dataset-local table assets such as:
  - `data/cleaned/<dataset_id>/tables/<paper_key>/*.csv`
  - `data/cleaned/<dataset_id>/tables/<paper_key>/tables_manifest.json`

**Function**

- Extract table-like assets from source documents.
- Preserve table order, caption/title references, source file identity, and raw row/column geometry as far as the extractor permits.

**Principles**

- Raw cleaned CSVs are legitimate inputs to S2-2 table recovery.
- Raw CSVs are not the long-term Stage5 value authority surface.
- If table extraction degrades geometry, S2-2 must either repair it or mark it explicitly as unrecoverable.

---

## S1-4 Manifest hydration / asset hydration

**Input**

- `manifest_current.tsv`.
- `key2txt.tsv`.
- Dataset-local table roots and table manifests.

**Output**

- Hydrated canonical manifest rows with text/table asset bindings.
- Explicit binding of:
  - `paper_key`
  - `text_path`
  - `text_source_type`
  - `pdf_path`
  - `html_path`
  - `table_asset_root`
  - source/dataset lineage fields.

**Function**

- Bind every manifest row to governed cleaned text and table asset references.

**Principles**

- Asset hydration is part of the manifest contract.
- Downstream stages should not discover text/table paths by directory search.
- Missing text/table bindings must be loud, not silent fallbacks.

---

## S1-5 Scope selection and run input contract

**Input**

- The one canonical `manifest_current.tsv`.
- Explicit selector parameters:
  - paper keys,
  - dataset tags,
  - split tags,
  - benchmark tags,
  - explicit scope TSV derived from the canonical manifest.

**Output**

- Run-scoped input contract, for example:
  - `RUN_CONTEXT.md`
  - `run_input_scope_v1.tsv`
  - `run_input_manifest_snapshot_v1.tsv`
  - `run_input_contract_v1.json`

**Function**

- Record the exact subset selected for a run.
- Record why each row is in scope.
- Record the text/table assets that Stage2 is allowed to consume.

**Principles**

- Subsets should be selected from the canonical manifest, not represented as competing manifests.
- Every run must carry its selected scope information.
- `paper_key`, `text_path`, and table asset references must be frozen before Stage2.

---

## S2-1 Scope resolution

**Input**

- Run input contract from S1-5.
- Canonical manifest row fields.
- Hydrated text/table bindings.

**Output**

- Per-paper scope context, preferably persisted as:
  - `semantic_stage2_objects/scope_context/<paper_key>/paper_scope_context_v1.json`

**Function**

- Resolve the per-paper Stage2 input boundary.
- Bind clean text, table assets, paper metadata, run lineage, and source lineage.

**Principles**

- This is resolution only.
- It does not rank, select, summarize, or semantically interpret evidence.
- Scope context must be exact enough for later audit to reproduce which source files were visible.

---

## S2-2a Candidate segmentation and full-table authority preservation

**Input**

- Per-paper scope context.
- Clean text.
- Raw table assets.

**Output**

- `semantic_stage2_objects/candidate_blocks/<paper_key>/candidate_blocks_v1.json`
- `semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`
- `semantic_stage2_objects/normalized_table_payloads/<paper_key>/payloads/*.csv`
- `table_cell_grid_v1.tsv/jsonl` or equivalent run-scoped table-cell grid.
- `analysis/candidate_segmentation_debug_v1.tsv`
- `analysis/table_authority_validation_v1.tsv`

**Function**

- Segment text/table candidate evidence.
- Recover and preserve execution-grade table authority.
- Preserve coordinate geometry, header hierarchy, row identity signals, and table-local source references.

**Principles**

- Only confirmed pure noise may be dropped irreversibly.
- Non-noise tables must remain preserved for authority even if not all are packed into prompts.
- Structural ranking is allowed for observability and authority-quality ordering; semantic table importance veto is not allowed here.
- This boundary preserves the complete table/value surface that downstream deterministic consumers must later reattach to.

---

## S2-2b Selector prioritization and evidence block construction

**Input**

- `candidate_blocks_v1.json`.
- `normalized_table_payloads_v1.json`.
- Feature activation/config.

**Output**

- `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
- `analysis/table_selection_debug_v1.json`
- selector recall / packing diagnostics.

**Function**

- Rank and pack semantic-facing evidence blocks.
- Ensure LLM sees high-value materials, preparation, formulation, table-summary, and result/characterization evidence within bounded context.
- Record selected-for-prompt vs preserved-for-authority status.

**Principles**

- Selector is a ranker/packer, not an authority filter.
- Selector must not define formulation rows or semantic truth.
- Evidence blocks may be compact and lossy, but must point back to full authority records where relevant.
- Evidence block construction must record feature activation, selection mode, technical status, and design status.

---

## S2-3 Prompt assembly

**Input**

- Persisted `evidence_blocks_v1.json` only.
- Prompt template/config.

**Output**

- In-memory prompt payload and/or:
  - `analysis/stage2_prompt_preview_v1.tsv`

**Function**

- Assemble semantic-facing prompt from frozen evidence blocks.

**Principles**

- Do not reread clean text.
- Do not rerank/rescore evidence.
- Do not place full numeric tables into prompt as execution authority.
- Prompt summaries should include identity/process/table-role semantic signals sufficient for the LLM to decide scope.

---

## S2-4a Prompt construction freeze

**Input**

- S2-3 prompt payload.

**Output**

- `analysis/s2_4a_prompt_template_v1.txt`
- `analysis/s2_4a_prompts_v1.jsonl`
- `analysis/s2_4a_prompt_audit_v1.tsv`
- Stage-local `RUN_CONTEXT.md`

**Function**

- Freeze prompt artifacts before any live LLM call.

**Principles**

- No live LLM call at this boundary.
- This is the cheapest audit point for prompt visibility and semantic adequacy.
- Any future live call must consume these frozen prompts or a newly governed prompt-freeze lineage.

---

## S2-4b Live LLM semantic discovery

**Input**

- Frozen prompts from S2-4a.
- Explicit model/backend/request metadata.

**Output**

- `raw_responses/<paper_key>__stage2_v2_raw_response.json`
- `request_metadata/`
- `analysis/s2_4b_request_summary_v1.tsv`
- Stage-local `RUN_CONTEXT.md`

**Function**

- LLM provides semantic signals: formulation-boundary signals, table-scope signals, preparation/process signals, relation/inheritance/selection cues, row-family meaning, and shared-vs-instance interpretation.

**Principles**

- LLM does not transmit concrete numeric table contents as the downstream value source.
- LLM does not own execution-side locators, table-cell coordinates, or final value materialization.
- LLM output is semantic authorization/signal, not the full table/value authority.

---

## S2-5 Semantic parsing

**Input**

- Frozen raw LLM responses.
- Minimal paper/run metadata.

**Output**

- `semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
- `semantic_stage2_objects/semantic_stage2_v2_summary.tsv`

**Function**

- Parse LLM responses into normalized semantic objects.
- Preserve semantic signals and evidence references.

**Principles**

- Parser must not add new semantic decisions.
- Parser must not silently drop table scopes, shared semantics, or relation cues.
- If raw response contains the signal but parsed objects do not, the failure is S2-5, not Stage5.

---

## S2-5b Semantic signal to S2-2 authority reattachment sidecar

**Input**

- S2-5 semantic objects.
- S2-2 full-table authority:
  - `normalized_table_payloads_v1.json`
  - payload CSVs
  - `table_cell_grid_v1.tsv/jsonl`
- S2-2 evidence block pointers and source locators.

**Output**

- Proposed explicit sidecar:
  - `semantic_stage2_objects/authority_reattachment/<paper_key>/semantic_authority_reattachment_v1.json`
  - run-level `analysis/semantic_authority_reattachment_audit_v1.tsv`

**Function**

- Convert LLM semantic signals into deterministic, auditable pointers back to S2-2 full authority surfaces.
- Resolve LLM-declared table/scope signals to stable table authority records.
- Provide downstream consumers with execution locators without asking the LLM to carry numeric rows.

**Principles**

- This is the critical bypass/sidecar rule previously designed: after S2-5, deterministic stages must be able to return to S2-2 and retrieve the complete table/value surface.
- Reattachment must be alias-aware but bounded:
  - logical table labels,
  - captions/titles,
  - source table references,
  - file-derived table IDs,
  - preserved authority rank/score.
- Reattachment must record ambiguity, conflicts, unresolved targets, and selected authority records.
- Reattachment must not invent semantic authorization; it only binds an existing semantic signal to preserved authority.

---

## S2-6 Contract validation

**Input**

- S2-5 semantic objects.
- S2-5b authority reattachment sidecar.
- Stage2 provenance and authority reports.

**Output**

- `analysis/stage2_semantic_authority_contract_report_v1.json`
- Stage-local `RUN_CONTEXT.md`

**Function**

- Validate Stage2 authority legality and provenance completeness.

**Principles**

- This is a legality/provenance gate, not selector logic.
- Validation should recognize lawful alias-equivalent reattachment surfaces.
- Failed reattachment should be reported as target-resolution failure, not silently projected downstream.

---

## S2-7 Compatibility projection

**Input**

- Passing S2-6 validation.
- S2-5 semantic objects.
- S2-5b authority reattachment sidecar.
- S2-2 table/value authority records.

**Output**

- `semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- `semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.jsonl`
- `semantic_to_widerow_adapter/compatibility_projection_trace_v1.tsv`
- `semantic_to_widerow_adapter/compatibility_projection_summary_v1.json`

**Function**

- Emit completed Stage2 candidate rows and compatibility fields for Stage3.
- Project LLM-authorized scopes into deterministic row-expansion units using reattached full-table authority.
- Preserve row-level binding surfaces such as `table_cell_bindings_json` where available.

**Principles**

- Completed weak-label TSV is the lawful Stage3 resume boundary.
- Raw responses and raw semantic objects are not lawful Stage3 inputs by themselves.
- Projection should expose whether row/value loss happened because of missing semantic signal, unresolved authority reattachment, or deterministic row expansion failure.

---

## S3-1 Relation materialization

**Input**

- Completed Stage2 weak-label TSV/JSONL.
- Scope/run metadata.

**Output**

- `formulation_relation_records_v1.tsv`
- `formulation_logic_graph_v1.jsonl`
- `formulation_relation_summary_v1.tsv`

**Function**

- Build explicit relation graph among formulation candidates.

**Principles**

- No LLM use.
- No rediscovery of missing chemistry from prose.
- Relation construction remains inside Stage2-authorized scope.

---

## S3-2 Relation resolution and shared carrythrough

**Input**

- Stage3 relation records / graph.
- Stage2 shared semantics and compatibility fields.

**Output**

- `resolved_relation_fields_v1.tsv`

**Function**

- Resolve parent/child inheritance, selected-condition inheritance, and unique shared preparation fields.

**Principles**

- Carrythrough must be source-backed, scope-aware, and entity-bound.
- No cross-table Cartesian reconstruction.
- If a value never reached Stage2 weak labels or shared semantic surfaces, Stage3 should not rediscover it.

---

## S4-1 Candidate diagnostics and review surfaces

**Input**

- Completed Stage2 candidate rows.
- Optional GT/review references.
- Scope/run metadata.

**Output**

- Candidate diagnostic summaries.
- Reviewer workbooks.
- Paper-level risk/audit queues.

**Function**

- Localize row-count, identity, and candidate-quality failures.

**Principles**

- Stage4 is diagnostic/reviewer-facing.
- Stage4 outputs are not benchmark-final and normally are not downstream execution inputs.

---

## S5-1 Fixed-row candidate intake

**Input**

- Completed Stage2 weak-label TSV.
- Stage3 relation records.
- Stage3 resolved relation fields.
- Scope/run metadata.

**Output**

- Fixed formulation-row universe for final closure.

**Function**

- Freeze the rows that Stage5 will close.

**Principles**

- Stage5 value layers attach values to admitted rows.
- Stage5 must not rediscover formulation membership.
- Later S5 substeps must not split/merge/create rows except under explicit governed identity rules.

---

## S5-2 Deterministic direct materialization

**Input**

- Fixed row universe.
- Stage2 compatibility row fields.
- S2-5b/S2-7 reattached authority locators and `table_cell_bindings_json`.
- Stage3 resolved relation fields.

**Output**

- Deterministic direct values in final-table columns.
- Decision trace / binding trace.

**Function**

- Materialize source-faithful direct values:
  - row-local table-cell binding,
  - multi-row header alignment,
  - group-label carry-down,
  - continuation-row merge,
  - direct value/unit split,
  - unique scoped preparation carrythrough.

**Principles**

- Consume S2-2 full-table authority through S2-5b/S2-7 locators; do not use prompt summaries as numeric authority.
- Do not donor-fill or assumption-fill.
- Do not convert concentrations into direct masses without derived-sidecar provenance.
- If S5-2 cannot materialize a value that exists in S2 authority, fix the deterministic consumer path before using S5-3.

---

## S5-3 LLM-assisted residual direct-value extraction

**Input**

- Fixed Stage5 rows.
- Audited residual direct-value gaps.
- Source/evidence artifacts already governed by upstream boundaries.

**Output**

- Direct-value candidates with evidence quote, scope, direct/derived classification, prompt hash, model identity, and validation status.

**Function**

- Fill residual source-evidenced gaps after S5-2 and Stage3-owned deterministic materialization have been exhausted.

**Principles**

- S5-3 is not database completion.
- S5-3 must not target blank schema slots merely because fields are empty.
- For DEV15, GT/error buckets can help define residual targets.
- For large-scale extraction without GT, S5-3 scope must be defined by source-observability signals, not GT gaps. See Part C repair plan.

---

## S5-4 Value authority validation and merge

**Input**

- S5-2 deterministic values.
- S5-3 candidate values if S5-3 was lawfully run.
- Direct/derived classification and provenance.

**Output**

- Accepted direct values.
- Rejected candidates / review queues.
- Merge decision trace.

**Function**

- Validate direct evidence, entity binding, scope, conflicts, and type compatibility.

**Principles**

- Row-local direct evidence outranks shared constants and LLM candidates.
- Ambiguous/conflict-bearing/quote-less values are rejected or sent to review.
- `present_but_mismatch` and `blocked_alignment` are separate review/alignment problems, not ordinary fill targets.

---

## S5-5 Derived reasoning sidecars

**Input**

- Accepted direct values.
- Formula/unit-conversion rules.

**Output**

- Derived sidecars with formula IDs, input provenance, and `eligible_for_direct_compare=no` unless explicitly governed otherwise.

**Function**

- Compute derived/calculated fields such as concentration × volume, unit conversions, ratio-derived masses, etc.

**Principles**

- Derived values do not replace direct fields.
- Derived values must not contaminate direct GT comparison.

---

## S5-6 Final table closure and audit export

**Input**

- Fixed row universe.
- Accepted direct values.
- Derived sidecars.
- Decision trace.

**Output**

- `final_formulation_table_v1.tsv`
- `final_output_decision_trace_v1.tsv`
- `final_output_summary_v1.md`
- Optional downstream/audit sidecars.

**Function**

- Emit one benchmark-facing row per formulation plus auditable sidecars.

**Principles**

- `final_formulation_table_v1.tsv` is the production endpoint.
- Stage5 final output is necessary but not sufficient for benchmark-valid reporting.

---

## P1 Identity freeze / diagnostic risk

**Input**

- `final_formulation_table_v1.tsv`.
- Upstream identity scaffold.

**Output**

- `identity_freeze_report_v1.tsv`
- `identity_freeze_summary_v1.tsv/md`

**Function**

- Detect row-count drift, identity reassignment, unresolved binding, and ambiguous scaffold binding.

**Principles**

- Identity risk makes the lineage diagnostic-only unless explicitly resolved.

---

## P2 GT compare / benchmark node

**Input**

- Stage5 final table.
- Frozen GT authority files.
- Declared scope.
- Identity freeze summary.

**Output**

- Count compare outputs.
- Layer3 value compare outputs when requested.

**Function**

- Compare final production output against locked manual reference assets.

**Principles**

- GT authority is locked separately.
- Intermediate outputs must not be reported as benchmark-final.
- Diagnostic compare remains `benchmark_valid=no`.

---

# Part B — Current Audit Findings, Separated From The Contract

## B1 Clean text visibility

Source diagnostic:

- `data/results/20260504_ab9f61e/062_source_anchor_cleantext_visibility_diagnostic/`

Observed summary:

```text
anchor_count                         15
anchor_visibility.full                6
anchor_visibility.partial             9
total_matched_fragments             333
total_exact_fragment_matches        231
total_numeric_token_fallback_matches 102
total_missing_fragments              37
anchor_visibility.absent              0
```

Interpretation:

- User-provided DEV15 source anchors are not wholly absent from clean text.
- However, 9/15 anchors are only partially visible, so source formatting/table-text preservation remains imperfect.

---

## B2 Table authority visibility

Source diagnostic:

- `data/results/20260504_ab9f61e/063_source_anchor_table_authority_visibility_diagnostic/`

Observed summary:

```text
anchor_count                         15
anchor_visibility.partial             8
anchor_visibility.absent              7
total_exact_fragment_matches         41
total_numeric_token_fallback_matches 244
total_missing_fragments             329
anchors_with_payload_json            15
anchors_with_grid_cells              15
absent_anchor_keys                   INMUTV7L,BB3JUVW7,BXCV5XWB,RHMJWZX8,WIVUCMYG,YGA8VQKU,5GIF3D8W
```

Interpretation:

- Payload/grid artifacts exist for all DEV15 anchors.
- Existence is not the same as exact evidence visibility.
- The first major repair target is Stage1/S2-2a table authority construction and exact table/text preservation.

---

## B3 Selector recall / authority-filter check

Source diagnostic:

- `data/results/20260504_ab9f61e/064_selector_anchor_recall_registry_diagnostic/`

Observed summary:

```text
candidate_rows                         812
selected_for_prompt                     59
preserved_for_authority                806
selector_authority_filter_violations     0
```

Interpretation:

- Current diagnostic surface did not find selector-as-authority-filter violations.
- Selector packing may still need better semantic adequacy, but not because it is obviously deleting authority.

---

## B4 Prompt summary semantic adequacy

Source diagnostic:

- `data/results/20260504_ab9f61e/065_prompt_summary_semantic_adequacy_diagnostic/`

Observed summary:

```text
evidence_blocks                                63
adequate                                       43
inadequate                                     20
first_failure.missing_identity_or_process_signal 20
```

Interpretation:

- Prompt summaries should not contain full numeric rows.
- The real prompt-side gap is missing identity/process/formulation-role signal in 20 blocks.

---

## B5 Unified first-failure classifier

Source diagnostic:

- `data/results/20260504_ab9f61e/066_stage1_stage2_visibility_boundary_classifier_diagnostic/`

Observed summary:

```text
paper_count                                         15
first_failure.none                                   3
first_failure.stage1_table_authority_visibility      7
first_failure.stage2_prompt_summary_semantic_adequacy 5
```

Interpretation:

- Current DEV15 source-anchor failures are mostly upstream of Stage5:
  - Stage1/S2-2a table authority visibility: 7 papers.
  - S2 prompt semantic adequacy: 5 papers.
  - No current visibility/summary first failure: 3 papers.

---

# Part C — Repair Plan

## Repair priority overview

1. Make canonical manifest + run scope contract explicit.
2. Strengthen Stage1/S2-2a table authority visibility and source preservation.
3. Implement/validate S2-5b semantic signal → S2-2 authority reattachment sidecar.
4. Improve S2-2b/S2-3 prompt semantic adequacy without requiring full numeric prompt tables.
5. Only then rerun S2 replay → S3 → S5 no-S5-3 diagnostic lineage.
6. Design a non-GT S5-3 scope policy for large-scale extraction, but do not run S5-3 until deterministic ownership gaps are closed.

---

## T01 — Document canonical manifest and run-scope contract

**Objective:** Ensure future runs use one authoritative manifest and record selected subsets as run input contracts.

**Files:**

- Create: `docs/plans/2026-05-06-end-to-end-boundary-repair-progress.tsv`
- Modify or create diagnostic helper only if needed under `src/stage1_cleaning/`.
- Do not modify `project/` unless separately approved.

**Steps:**

1. Define schema for `run_input_contract_v1.json` and `run_input_scope_v1.tsv` in this plan/progress ledger.
2. Add a diagnostic-only writer or validator if no existing maintained helper already records:
   - canonical manifest path,
   - selected paper keys,
   - selection rule/tag,
   - text paths,
   - table asset references.
3. Unit test that multiple raw source collections can map into the same canonical manifest and that scope selection is recorded as a run parameter, not a new competing manifest.

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest <new_or_existing_scope_contract_tests>
PYTHONPATH=. python3 -m py_compile <touched_files>
```

**Acceptance:**

- A diagnostic run can print and write the exact selected subset and source paths.
- No code path treats a subset manifest as a second authority competing with `manifest_current.tsv`.

---

## T02 — Add S2-1 paper scope context artifact

**Objective:** Persist the currently implicit per-paper Stage2 input boundary.

**Files:**

- Modify or extend Stage2 diagnostic/helper code under `src/stage2_sampling_labels/`.
- Tests under `tests/`.

**Steps:**

1. Write RED tests for `paper_scope_context_v1.json` containing:
   - `paper_key`,
   - canonical manifest path/hash,
   - run scope selection reason,
   - clean text path/source type,
   - PDF/HTML paths,
   - table asset roots/references.
2. Implement diagnostic writer without changing semantic extraction behavior.
3. Generate a bounded DEV15 diagnostic artifact.

**Verification:**

- Scope context exists for 15 DEV15 anchors.
- Every context points to existing clean text and records table asset availability or explicit absence.

---

## T03 — Repair Stage1/S2-2a table authority exact visibility audit into actionable buckets

**Objective:** Convert current 063 partial/absent findings into paper/table-level failure classes before modifying runtime behavior.

**Files:**

- Extend existing diagnostic helper:
  - `src/stage1_cleaning/audit_source_anchor_cleantext_visibility_v1.py`
  or create a Stage2-specific diagnostic under `src/stage2_sampling_labels/` if cleaner.
- Tests under `tests/`.

**Steps:**

1. For each anchor/table, classify missing exact visibility as one of:
   - raw table asset missing,
   - table asset exists but wrong table selected,
   - payload exists but row/header geometry degraded,
   - payload exists but text normalization mismatch,
   - table lives only in clean prose/inline text,
   - source excerpt is method/prose not expected in table payload.
2. Preserve numeric-token fallback as signal-only, never as visibility proof.
3. Write per-paper/table diagnostic TSV.

**Verification:**

- 7 table-authority absent anchors are assigned concrete first-failure classes.
- The diagnostic records exact payload/grid paths used.

**Acceptance:**

- We know whether the next runtime repair is Stage1 extraction, S2-2a reconstruction, alias rebinding, or prompt-only summary repair.

---

## T04 — Implement bounded S2-2a table-authority preservation repairs

**Objective:** Repair generic table-authority failures exposed by T03, not paper-specific final values.

**Potential runtime surfaces:**

- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
- shared table structure helpers if present.
- `src/stage2_sampling_labels/table_cell_grid_v1.py`

**Allowed generic repairs:**

- Preserve blank placeholder cells needed for header alignment.
- Strengthen multi-row header recovery.
- Preserve first-column row identity.
- Preserve caption/table source references.
- Add bounded alias handling across logical table labels, captions, and file-derived table IDs.
- Mark unusable/broken payloads explicitly instead of silently treating them as authority-complete.

**Prohibited repairs:**

- Paper-key branches.
- GT-count-driven row invention.
- Generic prose mining outside S2-2a recovery scope.
- Stage5 raw table backfill as a substitute.

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest <table_authority_tests>
PYTHONPATH=. python3 -m py_compile src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py src/stage2_sampling_labels/table_cell_grid_v1.py
```

Then rerun diagnostic-only table-authority visibility audit and compare against 063.

**Acceptance:**

- Exact table-authority visibility improves or failure classes become explicitly marked unrecoverable.
- No selector authority-filter violations are introduced.

---

## T05 — Implement S2-5b semantic signal to S2-2 authority reattachment sidecar

**Objective:** Make explicit the bypass/sidecar rule: LLM gives semantic signal; deterministic consumers return to S2-2 to retrieve full tables/values.

**Files:**

- Create or modify under `src/stage2_sampling_labels/`, e.g.:
  - `semantic_authority_reattachment_v1.py`
- Tests under `tests/test_semantic_authority_reattachment_v1.py`.

**Required behavior:**

- Input semantic table/scope signals from S2-5.
- Input S2-2 normalized payloads and table grid.
- Resolve authority target using bounded aliases:
  - logical table label,
  - caption/title,
  - source table reference,
  - file-derived table number,
  - authority score/rank.
- Output:
  - selected authority record,
  - confidence/ambiguity status,
  - candidate alternatives,
  - unresolved reason,
  - source paths.

**Tests:**

- Semantic `Table 1` resolves to higher-authority payload when exact low-authority asset is corrupted but caption/source alias matches.
- Ambiguous equal-authority aliases remain unresolved/review-required.
- No semantic signal means no authority target is created.
- Reattachment never copies full numeric rows into LLM semantic objects.

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_semantic_authority_reattachment_v1
PYTHONPATH=. python3 -m py_compile src/stage2_sampling_labels/semantic_authority_reattachment_v1.py
```

**Acceptance:**

- Downstream S2-7/S5-2 can consume stable authority locators instead of relying on prompt summaries or LLM-transmitted numeric values.

---

## T06 — Integrate reattachment sidecar into S2-6/S2-7 validation/projection diagnostics

**Objective:** Ensure unresolved authority reattachment is visible before Stage3/Stage5.

**Files:**

- `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py`
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
- Tests under `tests/`.

**Steps:**

1. Add diagnostic fields to validation report:
   - semantic signal count,
   - reattached target count,
   - unresolved target count,
   - ambiguous target count.
2. Add projection trace fields:
   - authority_target_id,
   - authority_payload_path,
   - table_cell_grid path or row ids,
   - reattachment_status.
3. Do not block all rows immediately unless the current contract requires it; first generate diagnostic-only visibility.

**Verification:**

- Compatibility projection traces expose whether row/value loss happened at semantic signal, reattachment, or deterministic expansion.

---

## T07 — Improve prompt summary semantic adequacy without adding full numeric tables

**Objective:** Address the 20 inadequate prompt evidence blocks caused by missing identity/process signal.

**Files:**

- Evidence-block/prompt summary code in `src/stage2_sampling_labels/`.
- Tests under `tests/test_prompt_summary_semantic_adequacy_v1.py` or companion test.

**Allowed repairs:**

- Include compact identity/process/formulation-role signals.
- Include table type/role summary.
- Include row identity surface summary.
- Include preparation/material/process local cues.
- Include pointer that full table authority exists and is preserved.

**Prohibited repairs:**

- Packing full numeric tables into prompts.
- Moving execution-side locators into semantic content as something the LLM must preserve.
- Reintroducing selector-as-authority filter.

**Verification:**

- Rerun prompt semantic adequacy diagnostic.
- Target: reduce `first_failure.missing_identity_or_process_signal` without increasing prompt/full-table authority confusion.

---

## T08 — Run no-new-LLM bounded Stage2 replay through S2-7

**Objective:** Validate deterministic repairs without spending live LLM calls.

**Inputs:**

- Frozen raw responses / accepted replay lineage, explicitly resolved.
- Canonical manifest and run scope contract.

**Outputs:**

- New diagnostic Stage2 replay child under `data/results/`.
- Stage2 validation report.
- Compatibility projection summary and trace.

**Verification:**

- Confirm request summary shows replay/no live LLM.
- Confirm S2-6 contract status and reattachment diagnostics.
- Compare row emission and projection failures against prior diagnostic baseline.

**Acceptance:**

- Improved authority reattachment and prompt adequacy signals without collapsing row universe.

---

## T09 — Run Stage3 and Stage5 no-S5-3 diagnostic replay

**Objective:** Test whether upstream repairs reduce Stage5 deterministic gaps before any S5-3 expansion.

**Inputs:**

- Completed Stage2 replay artifact from T08.
- Stage3 maintained entrypoint.
- Stage5 final table builder with S5-3 disabled/not invoked.

**Outputs:**

- Stage3 relation artifacts.
- Stage5 `final_formulation_table_v1.tsv`.
- Decision trace.
- Identity freeze diagnostics if compare is run.

**Verification:**

- Final row count does not regress unexpectedly.
- Stage5 decision trace shows values consumed from S2/S3 authority locators where available.
- No raw-source Stage5 mining is introduced.

---

## T10 — Compare diagnostic lineage and classify residuals by first-failure boundary

**Objective:** Ensure remaining errors are assigned to the right owner.

**Inputs:**

- T09 final table.
- Frozen GT authority paths from `ACTIVE_RUN.json` or explicit contract.
- Identity freeze diagnostics.

**Outputs:**

- Diagnostic compare outputs.
- Residual classifier TSV with buckets:
  - source absent from clean text,
  - table authority absent/degraded,
  - selector/prompt semantic inadequacy,
  - LLM did not authorize,
  - parser dropped signal,
  - authority reattachment unresolved,
  - S2-7 projection failed,
  - Stage3 carrythrough failed,
  - S5-2 deterministic consumer failed,
  - lawful S5-3 residual direct-value candidate,
  - not reported/no source,
  - blocked alignment,
  - present but mismatch.

**Principles:**

- Only the `lawful S5-3 residual direct-value candidate` bucket is a candidate for S5-3.
- `not_reported/no source`, `blocked_alignment`, and `present_but_mismatch` are not S5-3 ordinary fill targets.

---

## T11 — Design non-GT S5-3 scope policy for large-scale extraction

**Objective:** Define how S5-3 can be scoped when no GT table exists.

**Problem:**

- DEV15 can use GT compare gaps to identify S5-3 residuals.
- Large-scale extraction has no GT table; blank final-table fields alone are not valid S5-3 targets.

**Proposed non-GT scope signals:**

A row/field may enter S5-3 only if at least one source-observability trigger exists:

1. Upstream evidence explicitly indicates the field is reported but S5-2 could not materialize it.
2. Evidence block/prompt summary contains a source-backed mention of the value type for the admitted row.
3. S2-2 table/grid has an unmapped row-local cell whose header maps to the target field but S5-2 rejected/failed it.
4. Stage3 resolved relation has a source-backed shared value but S5-2 failed final-field promotion.
5. Compare-free internal consistency detects a typed direct value candidate with quote/locator but no final-field assignment.

Explicit exclusions:

- Field blank with no source-observability signal.
- Paper-level field never reported.
- Ambiguous source scope.
- Derived-only candidate for direct field.
- Row identity/alignment unresolved.
- Value only present in unrelated results/assay/control context.

**Output:**

- `s5_3_scope_candidates_v1.tsv` with:
  - `paper_key`,
  - `formulation_id`,
  - `field_name`,
  - `scope_trigger`,
  - `source_locator`,
  - `upstream_boundary`,
  - `why_s5_2_failed`,
  - `eligible_for_s5_3`,
  - `exclusion_reason`.

**Acceptance:**

- S5-3 target scope is source-observability-driven, not GT-driven and not blank-schema-slot-driven.

---

## T12 — Update progress ledger and request review before runtime promotion

**Objective:** Keep the work resumable and governed.

**Files:**

- `docs/plans/2026-05-06-end-to-end-boundary-repair-progress.tsv`

**Steps:**

1. After each task, update status, artifact paths, test status, and blockers.
2. Use read-only specialist review for nontrivial repairs.
3. Do not update `ACTIVE_RUN.json` unless a full diagnostic lineage is validated and explicitly selected.

---

# Part D — Initial Progress Ledger Schema

Create `docs/plans/2026-05-06-end-to-end-boundary-repair-progress.tsv` with columns:

```text
task_id	status	owner	primary_files	artifacts	tests	review_status	updated_at	notes
```

Initial statuses:

```text
T01	pending
T02	pending
T03	pending
T04	pending
T05	pending
T06	pending
T07	pending
T08	pending
T09	pending
T10	pending
T11	pending
T12	pending
```

---

# Part E — Expected Repair Order

1. Scope/manifest contract diagnostics.
2. Table authority failure-class diagnostics.
3. S2-2a generic table authority repairs.
4. S2-5b authority reattachment sidecar.
5. S2-6/S2-7 validation/projection integration.
6. Prompt semantic adequacy repair.
7. No-new-LLM Stage2 replay.
8. Stage3/Stage5 no-S5-3 replay.
9. Residual first-failure classifier.
10. Non-GT S5-3 scope policy.

This order is intentional: it prevents repeating the historical failure mode where Stage5 is repeatedly patched even though the first loss happened in clean text, table authority, selector/prompt, or semantic-signal reattachment.
