# S2-2 Pre-LLM Artifact Surface Audit

Date: `2026-04-10`

Scope audited: clean text -> evidence blocks / pre-LLM evidence package only.

Files read for authority before this audit:
- `AGENTS.md`
- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/PIPELINE_SCRIPT_MAP.md`
- `project/feature_units/FEATURE_UNIT_GOVERNANCE.md`
- `project/feature_units/feature_unit_registry.json`
- `project/feature_units/feature_intervention_matrix.tsv`
- `docs/maintained_script_surface.tsv`

Memory bootstrap query used:
- `python src/utils/mem_bootstrap_v1.py --query "S2-2 clean text evidence blocks pre-LLM evidence package stage2_prompt_preview table_selection_debug text_path manifest key2txt"`

Strictness note:
- This report lists only artifacts, fields, and producer/consumer links verified from current repo code or current on-disk artifacts.
- Historical Stage2.5A and legacy fixparse materials were found in memory and docs, but they are not treated here as the maintained S2-2 runtime unless current maintained code still emits or consumes them.

## Current S2-2 Producer Chain

1. Stage1 authoritative manifest:
- `src/stage1_cleaning/zotero_raw_to_manifest.py` produces `data/cleaned/index/manifest_current.tsv`.
- Actual fields in `manifest_current.tsv`: `key`, `zotero_key`, `title`, `doi`, `year`, `pdf`, `html`, `notes`.

2. Stage1 clean-text production:
- `src/stage1_cleaning/clean_manifest_to_text.py` is the maintained wrapper.
- It runs `src/stage1_cleaning/pdf2clean.py`.
- It promotes a normalized two-column mapping into `data/cleaned/index/key2txt.tsv`.
- The wrapper comments define `data/cleaned/index/key2txt.tsv` as the authoritative key-to-text mapping.

3. Stage1 richer cleaner summary:
- `src/stage1_cleaning/pdf2clean.py` writes `data/cleaned/content/key2txt.tsv`.
- Actual fields observed there: `key`, `title`, `source_type`, `txt_path`, `text_length`, `table_detected`, `parse_quality`, `notes`, `page_count`, `url`.
- This is richer than the maintained two-column index mapping, but the wrapper explicitly treats it as an intermediate that gets normalized.

4. Stage2 maintained entrypoint:
- `src/stage2_sampling_labels/run_stage2_composite_v1.py` is the one maintained Stage2 entrypoint.
- It writes a run-local `targeted_manifest.tsv` and passes that file into `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`.
- The extractor argument contract is explicit: `--manifest-tsv` must contain `key/doi/title/text_path` columns.

5. Stage2 actual pre-LLM assembly:
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py` resolves `text_path` from the manifest row.
- It resolves tables from `text_path.parent.parent / "tables" / key` first, then dataset fallbacks under `data/cleaned/content_goren_2025/tables/<key>` and `data/cleaned/goren_2025/tables/<key>`.
- In `live_llm` mode it builds the prompt in memory through `build_live_prompt(...)`.
- If `STAGE2_INPUT_EVIDENCE_PACKING_MODE=ordered_blocks`, `build_live_prompt(...)` calls `build_controlled_evidence_pack(...)`.
- Otherwise the maintained fallback is a raw text prefix plus rendered table excerpts.

6. Persisted maintained observability at this boundary:
- `analysis/stage2_prompt_preview_v1.tsv` is emitted for live Stage2 runs.
- `analysis/table_selection_debug_v1.json` is emitted only when `STAGE2_TABLE_MODE=summary`.
- `RUN_CONTEXT.md` records the Stage2 input settings.
- `analysis/feature_activation_report_v1.tsv` can mark `stage2_input_evidence_packing` active by reading the prompt preview.

7. Important repo-real limitation:
- No maintained script in the audited Stage1 set writes a canonical Stage2-ready scope manifest with `text_path` as part of the Stage1 authoritative output surface.
- The maintained Stage2 code clearly requires such a manifest, and a real example exists at `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`, but this audit did not identify a maintained Stage1 producer for that file.

## Current Artifact Inventory

### Upstream maintained clean-text inputs

- `data/cleaned/index/manifest_current.tsv`
  - Producer: `src/stage1_cleaning/zotero_raw_to_manifest.py`
  - Status: maintained upstream input
  - Granularity: corpus-level
  - Stage2 direct consumption: no, not by itself, because current Stage2 requires `text_path`

- `data/cleaned/index/key2txt.tsv`
  - Producer: `src/stage1_cleaning/clean_manifest_to_text.py`
  - Status: maintained upstream input
  - Granularity: corpus-level
  - Stage2 direct consumption: not directly by the extractor code audited here
  - Function: authoritative key -> cleaned text path mapping

- `data/cleaned/content/key2txt.tsv`
  - Producer: `src/stage1_cleaning/pdf2clean.py`
  - Status: supporting Stage1 summary
  - Granularity: corpus-level
  - Stage2 direct consumption: none found
  - Function: richer cleaner status table with `txt_path`, parse quality, and related metadata

- Cleaned text files under the paths referenced by `text_path`
  - Producer: `src/stage1_cleaning/clean_manifest_to_text.py` via `src/stage1_cleaning/pdf2clean.py`
  - Actual observed paths include `data/cleaned/content_goren_2025/text/<KEY>.pdf.txt` and `.html.txt`
  - Status: maintained upstream input
  - Granularity: per-paper
  - Stage2 direct consumption: yes

- Stage2 scope manifest with `text_path`
  - Example verified file: `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`
  - Status: current real input surface, but producer not identified in the maintained Stage1 code read for this audit
  - Granularity: per-run or per-scope
  - Stage2 direct consumption: yes
  - Function: concrete carrier of `key`, `doi`, `title`, and `text_path` into the extractor

### Maintained live pre-LLM observability

- `analysis/stage2_prompt_preview_v1.tsv`
  - Producer: `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  - Status: supporting internal artifact auto-emitted by the maintained live Stage2 path
  - Granularity: per-run, one row per selected paper
  - Downstream consumption:
    - consumed by `src/utils/build_feature_activation_report_v1.py`
    - consumed by `src/utils/build_feature_execution_ledger_v1.py`
    - not consumed by Stage3/Stage5 runtime
  - Function: run-local audit surface for actual prompt shape, layout class, block ordering summary, and preview text

- `analysis/table_selection_debug_v1.json`
  - Producer: `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  - Status: supporting internal artifact auto-emitted only when summary table mode is active
  - Granularity: per-run with per-document payloads
  - Downstream consumption:
    - consumed by `src/utils/build_feature_execution_ledger_v1.py`
    - no maintained Stage3/Stage5 runtime consumer found
  - Function: debug surface for ranked summary-table selection

### Supporting-only evidence bundle path

- `stage2_validation/evidence_bundles_v1.jsonl`
  - Producer: `src/stage2_sampling_labels/build_evidence_bundle_for_keys_v1.py`
  - Status: supporting-only `STABLE_TOOL`, not the maintained Stage2 entrypoint
  - Granularity: per-run, one JSON object per paper
  - Downstream consumption:
    - consumed by `src/stage2_sampling_labels/export_evidence_bundle_audit_xlsx_v1.py`
    - not consumed by maintained Stage2/Stage3/Stage5 runtime
  - Function: explicit text-plus-selected-tables audit bundle

- `stage2_validation/evidence_bundles_v1_summary.tsv`
  - Producer: `src/stage2_sampling_labels/build_evidence_bundle_for_keys_v1.py`
  - Status: supporting-only
  - Granularity: per-run, one row per paper
  - Downstream consumption: no maintained runtime consumer found
  - Function: compact summary of bundle completeness

- `evidence_audit_v1.xlsx`
  - Producer: `src/stage2_sampling_labels/export_evidence_bundle_audit_xlsx_v1.py`
  - Status: supporting-only
  - Granularity: per-run workbook
  - Downstream consumption: human audit only
  - Function: workbook view over `evidence_bundles_v1.jsonl`

### Historical or unattributed artifacts found

- `data/results/20260407_ab12cd3/18_qlyk_input_contract_live_validation/analysis/qlyk_table_selector_scoring_v1.json`
  - Producer: not found by current repo search for the filename or token
  - Status: artifact exists on disk, but no current producer script was identified
  - Granularity: per-run
  - Downstream consumption: none found
  - Function: run-local table scoring debug surface only

- `src/stage2_sampling_labels/export_blockpack_audit_v7pilot_r3_fixparse.py`
  - Current dependency: imports the deprecated legacy extractor `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
  - Status: supporting/historical diagnostic only, not part of the maintained Stage2 composite
  - Function: manual audit export for legacy block packing, not current maintained S2-2

## Current Field Inventory

### Upstream inputs actually present

- `data/cleaned/index/manifest_current.tsv`
  - `key`
  - `zotero_key`
  - `title`
  - `doi`
  - `year`
  - `pdf`
  - `html`
  - `notes`

- `data/cleaned/index/key2txt.tsv`
  - no header
  - actual two columns only:
  - column 1: key
  - column 2: repo-relative cleaned text path

- `data/cleaned/content/key2txt.tsv`
  - `key`
  - `title`
  - `source_type`
  - `txt_path`
  - `text_length`
  - `table_detected`
  - `parse_quality`
  - `notes`
  - `page_count`
  - `url`

- Real Stage2 scope manifest example:
  - verified file: `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`
  - actual fields observed:
  - `key`
  - `zotero_key`
  - `paper_id`
  - `title`
  - `doi`
  - `year`
  - `text_path`
  - `selection_reason`
  - `pilot_reason`

### Maintained live pre-LLM surfaces

- `analysis/stage2_prompt_preview_v1.tsv`
  - actual written fields from code and verified on disk:
  - `document_key`
  - `doi`
  - `source_mode`
  - `table_mode`
  - `summary_first_column_enhancement`
  - `input_packing_mode`
  - `prompt_layout_class`
  - `paper_text_marker_index`
  - `evidence_pack_marker_index`
  - `table_excerpts_marker_index`
  - `ordered_block_order`
  - `prompt_length`
  - `prompt_head_preview`
  - `prompt_tail_preview`

- `analysis/table_selection_debug_v1.json`
  - top-level per-document fields verified on disk:
  - `document_key`
  - `table_mode`
  - `selection_ranking_mode`
  - `summary_first_column_enhancement`
  - `max_tables`
  - `selected_tables`
  - `candidates`
  - candidate fields verified on disk:
  - `file`
  - `score`
  - `selected`
  - `row_pattern`
  - `page_number`
  - `n_rows`
  - `n_cols`
  - `caption_or_title`
  - `header_keywords_hit`
  - `first_data_row_preview`

### Supporting-only evidence bundle fields

- `stage2_validation/evidence_bundles_v1.jsonl`
  - `key`
  - `doi`
  - `source_text_path`
  - `source_text`
  - `preferred_table_source`
  - `selected_table_files`
  - `selected_tables_tsv`
  - `notes`

- `stage2_validation/evidence_bundles_v1_summary.tsv`
  - `key`
  - `has_text`
  - `n_selected_tables`
  - `preferred_table_source`
  - `missing_table_manifest`
  - `missing_text`

- `data/results/20260407_ab12cd3/18_qlyk_input_contract_live_validation/analysis/qlyk_table_selector_scoring_v1.json`
  - actual fields observed on disk:
  - `file`
  - `score`
  - `row_pattern`
  - `role_hint`
  - `n_rows`
  - `n_cols`
  - `fraction_numeric_cells`
  - `header_keywords_hit`
  - `caption_or_title`
  - `first_row`
  - `sample_row_1`

### Stage2 settings and activation fields relevant to this boundary

- `RUN_CONTEXT.md`
  - verified Stage2 setting lines:
  - `stage2_table_mode`
  - `stage2_table_summary_first_column_enhancement`
  - `stage2_input_evidence_packing_mode`

- `analysis/feature_activation_report_v1.tsv`
  - verified fields:
  - `feature_id`
  - `expected_for_run`
  - `observed_activation`
  - `activation_status`
  - `activation_state`
  - `evidence_path`
  - `evidence_detail`
  - `notes`

## Current Success Signal Inventory

### Technical success signals

- `analysis/stage2_prompt_preview_v1.tsv`
  - This is the only maintained run-local proof that a live pre-LLM prompt was assembled.
  - It proves layout class and input-packing mode for the live path.
  - It does not persist the full prompt text or the full evidence pack body.

- `analysis/table_selection_debug_v1.json`
  - This is a real technical trace for summary-mode table ranking and selection.
  - It is conditional.
  - It is absent in full-table mode.

- `RUN_CONTEXT.md`
  - This records requested Stage2 input settings.
  - It is configuration evidence, not proof that the live prompt actually used those settings.

### Feature activation signals

- `analysis/feature_activation_report_v1.tsv`
  - Maintained run-level activation surface.
  - `src/utils/build_feature_activation_report_v1.py` reads `stage2_prompt_preview_v1.tsv` to observe `stage2_input_evidence_packing`.
  - It treats missing prompt preview as missing Stage2 live visibility.

- `RUN_CONTEXT.md` `## Feature Unit Activation`
  - Maintained summary integration written by `src/utils/update_run_context_with_feature_activation_v1.py`.
  - Real example verified in `data/results/20260401_5d9f4e6/17_ordered_packing_smoke5/RUN_CONTEXT.md`.

### Design-conformance-like signals

- `src/utils/build_feature_execution_ledger_v1.py`
  - Supporting-only governance utility.
  - It reads `analysis/stage2_prompt_preview_v1.tsv` and `analysis/table_selection_debug_v1.json`.
  - It can classify `stage2_live_visibility` features such as ordered packing, prompt preview observability, summary mode, and raw-prefix fallback.
  - It is not auto-run by the maintained Stage2 wrapper.

- `src/utils/build_feature_governance_signal_v1.py`
  - Supporting-only governance utility.
  - It depends on the execution ledger.
  - It is not auto-run by the maintained Stage2 wrapper.

- `docs/feature_governance/feature_applicability_schema_v1.tsv`
  - Supporting documentation and adjudication layer.
  - It records blocking/secondary/observability tiers for this boundary.
  - It is not itself a run-local success artifact.

## Gaps Relative To A Strict S2-2 Contract

These are repo-real gaps, not redesign proposals.

- No maintained persisted artifact stores the actual pre-LLM evidence pack body.
  - `build_controlled_evidence_pack(...)` exists in maintained code, but its output is only carried into the prompt string in memory.
  - The maintained run-local artifact is a preview row, not the evidence pack itself.

- No maintained persisted artifact stores per-block evidence contents or per-block spans for the live Stage2 prompt builder.
  - The only persisted ordering field is `ordered_block_order` in the prompt preview.

- No maintained persisted artifact stores table scores or selected table IDs in full-table mode.
  - `table_selection_debug_v1.json` exists only in summary mode.

- No maintained Stage1 producer was identified in this audit for the Stage2-ready manifest that includes `text_path`.
  - `manifest_current.tsv` does not include `text_path`.
  - `key2txt.tsv` carries the mapping, but the current maintained Stage2 extractor requires `text_path` inside its manifest rows.
  - Real scope manifests with `text_path` exist, but their maintained upstream producer was not identified in the scripts read here.

- `RUN_CONTEXT.md` records requested input settings but is not enough to prove live prompt conformance.
  - The prompt preview is the stronger technical proof.

- The feature-unit registry covers `stage2_input_evidence_packing`, but current repo materials outside the registry identify additional S2-2 boundary features such as summary mode and prompt preview observability.
  - Those are visible in supporting governance docs and ledger code, not in the maintained feature-unit registry used for the default activation report.

- Replay visibility is limited.
  - Supporting governance code explicitly treats prompt preview visibility as potentially hidden upstream for replay runs.

- `qlyk_table_selector_scoring_v1.json` exists as a real run artifact, but this audit did not find a current producer script for it.
  - It therefore cannot be treated as a maintained current contract surface.
